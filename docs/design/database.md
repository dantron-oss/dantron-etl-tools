# Database: connections and loading rows

Notes on `db.py`: how a step reaches a database, and how rows are written. Each heading is one
decision: what was decided, why, and what was rejected.

## One module for every table tool

**Decided.** Reading the environment, picking the driver by scheme, and refusing an unsafe name
before a connection opens are the same steps for a tool that reads a table and one that writes one.
They are written once, in `db.py`.

**Why.** A new tool that reaches for a database then gets the same refusals instead of inventing a
slightly different set.

## Credentials come from the environment

**Decided.** A step names an environment variable. The connection string is read from `os.environ`
and nowhere else. A `.env` file is never parsed.

**Why.** A pipeline file is committed and handed to another client as a template, so it must never
hold a credential. In an orchestrator or in CI there is no `.env` file: the secret arrives from a
secret store. A library that parsed one would bake one project's layout into every consumer.

**Messages never repeat a secret.** A setting that does not look like a variable name is probably the
connection string itself, written where the variable name should be. It is refused without being
quoted. A variable that is not set is refused by its name, which is safe to repeat.

**Driver errors are not shown.** A driver's message can quote the connection string, password
included. A failed Postgres connection reports the database, host, port and the kind of error, and
the refusal is raised outside the `except` block so the driver's exception is not chained to it. A
traceback, a trace or a log cannot show the driver's message.

## A plain name is the only guard

**Decided.** `schema` and `table` must be plain identifiers: letters, digits and underscores, not
starting with a digit. `validate_identifier` checks this before any connection opens. The pattern is
matched with `fullmatch`, because a pattern ending in `$` also accepts a name with a trailing newline.

**Why.** A table or schema name is written into the SQL text. It cannot be passed as a bound
parameter. Refusing anything but a plain name is the only defence against a name that carries a
second statement, and it only works if it runs first.

Column names read back from the database itself are quoted instead (`quoted_identifier`), because
they come from the database's catalog, not from a pipeline file. Quoting also keeps a column named
like a reserved word (`order`) or holding a space readable. The spelling is the same on both
drivers.

## No driver is declared

**Decided.** The package declares no database driver. SQLite uses the standard library. A Postgres
connection imports `psycopg` only when one is asked for, and falls back to ADBC
(`adbc_driver_postgresql`). With neither installed, the refusal names the packages to install.

**Rejected.** Letting the bare `ImportError` surface. It names a library the caller has never heard
of.

## A missing SQLite file is refused

**Decided.** `sqlite:///<path>` with a path that does not exist is refused. A `file:` URI that states
no `mode` is opened with `mode=rw`, so it must exist too. A URI that states a mode is opened as
written.

**Why.** `sqlite3.connect` creates a missing file. A read or a write against the new empty file then
reports success having touched nothing the author meant, or reports a missing table while leaving an
empty database behind.

## Dates in SQLite

**Decided.** The module registers its own SQLite adapters that store `date` and `datetime` as ISO
text.

**Why.** sqlite3's built-in adapters for these types were deprecated in Python 3.12 and are removed
in 3.14. Without our own, a `Date` column would warn today and fail on a later Python.

## One type vocabulary for both engines

**Decided.** `declared_type_of` reads a SQL type name into the library's column type names (`Int64`,
`Float64`, `String`, `Boolean`, `Date`, `Datetime`, `Time`). SQLite gives the word from the table
definition; Postgres gives `format_type`'s spelling. Both go through the same table, so a frame
compared with a table gets the same answer whichever engine holds the table.

Width, precision and time zone are dropped before the lookup. `varchar(20)` and `timestamp without
time zone` say nothing about which column type a value must be, and keeping them would answer two
tables declared slightly differently in different ways. `numeric` and `decimal` read as `Decimal`;
their digits and places are read separately, by `numeric_declaration`, because they decide which
values fit (see below).

**A name not in the table gets `None`, meaning "no opinion", not "wrong".** A domain, an enum, a JSON
column, an array and any unknown name are passed over. Any doubt costs the slower answer, never a
wrong one. An array is not reduced to its element type, because a column of arrays holds something
no column type here describes.

## Numbers into a numeric column

**Decided.** A number goes into a `numeric(p,s)` column only when every value fits s places and
p - s digits before the point, checked before anything is written, on both engines. An exact column
is read as it is; a whole number too; a Float64 is read to 15 significant digits (a Float32 to 6),
the reading `round` and `cast_columns` use, and that reading is what is sent, so the database
stores exactly what was checked. A negative scale (`numeric(5,-2)`) holds whole hundreds only. A
`numeric` with no places stated takes any number as it is. An exact column never goes into a
`double precision`, `real` or `float` column.

**Why.** Postgres rounds a value with more places than the column holds, half away from zero,
without a word: 1.235 into `numeric(5,2)` is stored as 1.24, and 10.255 sent as a float is stored
as 10.26. That is a figure changed by a write, which the owner's rulings forbid: nothing rounds
unless the author asked. An exact column into a float column would keep only the nearest binary
value, which is the exact column meeting a float, refused everywhere else in the library. A float
into a float column is the path the first project publishes on, and it is untouched.

**SQLite.** It has no exact type and its driver cannot send a Python `Decimal`, so the library
registers an adapter that sends the digits as text. A column declared `numeric` then stores them by
SQLite's own rule (a REAL or an INTEGER). The declared `numeric(p,s)` is still checked, because
the declaration is what the table promises whoever reads it.

## Reading a table's rules

**Decided.** `TableRules` records what a table demands of a row, as far as the catalog can be read
and the rows can be judged: not-null columns, one-column unique rules, and the flag
`beyond_the_frame`.

`beyond_the_frame` is true whenever the table has a rule the rows in hand cannot be checked against:
a CHECK, a foreign key, a unique rule or key over several columns, an exclusion constraint, a unique
index with no constraint behind it. It defaults to true. A reader that only half understands a table
says so, instead of reporting the rules it recognised as if they were all of them.

`settled_by_the_engine` names columns where a null is not a value to judge, because the engine
decides what goes there. SQLite fills a null in a rowid-alias key with a number of its own, which a
later row can then collide with, and accepts a null in any other key more than once. Neither is
visible in the rows as written.

The `table_rules`, `declared_column_types` and `value_types` methods of `RowLoader` are
optimisations. Their default answers (cannot judge; no opinion) cost a slower diagnosis or a skipped
check, never a wrong refusal. Every other method of `RowLoader` raises `NotImplementedError` by
default, because a load cannot go ahead without it.

### SQLite CHECK rules

The pragmas give nullability and the key (`table_info`), unique indexes and whether a constraint
stands behind each (`index_list`, `index_info`), and foreign keys (`foreign_key_list`). A CHECK
appears in none of them; it exists only as text in the definition. Rather than parse SQL, a table
whose definition contains the word `check` anywhere is declared unjudgeable. That over-reaches on a
column merely named "check", which costs a slower diagnosis, never a wrong one.

In `table_info`, `pk` is the column's position in the primary key (counting from one, zero outside
it), so it answers "is this column in the key" and "does the key span columns" at once.

### SQLite declared types

`cursor.description` cannot answer this on SQLite: its type codes are absent unless the connection
uses the detect-types modes. The storage class a value comes back as belongs to the value, not the
column: SQLite stores the text `'heavy'` in a column declared `integer` and reports success. The
declaration in `table_info` is the only thing that says what the column was meant to hold.

### Postgres declared types

`cursor.description` carries type codes on Postgres, but they are OIDs that would have to be looked
up in the catalog anyway, and a domain gives its own OID, not the type underneath. Asking the
catalog with `format_type` is the same round trip and gives the name the definition was written
with.

### Postgres rules

Three catalog reads, none against the table itself: `pg_attribute` for not-null, `pg_constraint` for
keys and other rules, and `pg_index` for a unique index no constraint stands behind, which enforces
uniqueness just as firmly and appears in neither of the first two. The constraint query turns column
numbers into names in the server, so a one-column key arrives under the name the frame uses. The
not-null constraint newer servers record in `pg_constraint` is skipped there, because
`attnotnull` has already answered it.

## One loader per driver

**Decided.** `RowLoader` holds one driver's answers to the questions a load asks that the standard
Python database interface (DB-API) does not settle: how a batch of rows is written, how a
transaction is opened, how one row is tried and undone, and which exceptions mean "the rows were
wrong" rather than "the run went wrong". A tool asks `driver_for(connection)` for the loader that
belongs to the connection it was handed.

**Why.** Every one of these differs between SQLite and Postgres. A tool that used the conveniences
of the driver it was written against would work on that driver only.

## Postgres loading

- **`COPY ... FROM STDIN`, not a row at a time.** A publish can be a hundred thousand rows. A
  statement per row is not a slower way to do that; it is a different job.
- **The driver's own row writer.** The `COPY` stream is tab-delimited and backslash-escaped. A value
  holding a tab, a newline or the two characters `\N` lands in the wrong column, or as a null,
  without any error if the escaping is done by hand.
- **`with connection:` is not a transaction on psycopg.** On psycopg it closes the connection; on
  sqlite3 it only commits. The Postgres loader uses `connection.transaction()`.
- **A failed statement spoils the whole transaction on Postgres.** Replaying rows one at a time to
  find the bad ones needs a savepoint per row. On SQLite a failed statement leaves the rest of the
  pass usable, so there is nothing to undo.

## Merging through a staging table

**Decided.** A merge loads the rows into a temporary table, `_etl_merge_staging`, then upserts from
it into the target in one `INSERT ... SELECT ... ON CONFLICT` statement.

**Why.** The bulk path on Postgres is `COPY`, which cannot resolve a conflict itself. Loading into a
staging table and upserting from it keeps the bulk load and gives both engines the same answer.
**Rejected:** a per-row upsert on one engine and `COPY` on the other, which would not.

**The name.** One fixed name, dropped as soon as the upsert has read it, so two merges in the same
transaction do not collide. It is spelled unlike anything a schema would hold because on SQLite a
temporary table hides a permanent one of the same name, and the merge would then read its own
staging table as the target.

**`WHERE true`.** SQLite refuses `INSERT ... SELECT ... ON CONFLICT` without a `WHERE` on the select,
because it reads `ON` as the start of a join. Postgres accepts the same clause, so one statement
serves both.

**Nulls.** With `keep_published`, each column is set to `coalesce(excluded.c, target.c)`: a value
overwrites, a null leaves the published value standing. Without it, `= excluded.c` writes the null.
The two agree on three of the four cases and differ on the one that matters, a value that has gone
blank upstream, so the caller must say which it wants.

### Merging in SQLite

SQLite has neither `CREATE TABLE ... (LIKE ...)` nor `ON COMMIT DROP`. The staging table's shape
comes from a select that returns no rows, and it is dropped by name. It is dropped before it is
created as well as after it is read, because a temporary table can outlive the transaction that made
it: a publish that failed part way would leave one behind, and the next attempt would fail on it for
a reason that has nothing to do with the rows.

### Merging in Postgres

`LIKE` copies the target's `NOT NULL` declarations, so a row the target would refuse is refused on
the way into the staging table instead of surviving to the upsert. `ON COMMIT DROP` removes the
table however the transaction ends. The explicit drop afterwards lets a second merge in the same
transaction reuse the name.

## An unknown driver is refused

**Decided.** `driver_for` checks what the connection is: a `sqlite3` connection or a `psycopg`
one. Anything else is refused by name.

**Rejected.** Recognising one driver and assuming the other. A connection from a third driver
answers most of the methods a load uses, so assuming would not fail loudly: it would write through
transaction and escaping rules nothing here has checked. This includes ADBC, which `open_connection`
falls back to when psycopg is missing; the refusal says to install psycopg to write to Postgres.

## Small limits

- SQLite statements carry at most 500 values each when checking which values a column already
  holds. SQLite's own limit is higher; staying under it costs one extra query per 500 values.
