# Changelog

What changed in each release, for people who use the library. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version (0.2, 0.3) may rename
things; each such release says how to upgrade.

## [Unreleased]

### Added

- **`Decimal` is a type name**, for exact decimals, with `places` (0 to 38) stated beside it. It is
  listed in `COLUMN_TYPES`.
- **`cast_columns` converts to `Decimal`**, written `{column: price, type: Decimal, places: 2}`. Text
  written as plain digits, whole numbers, exact columns of other places and `Float64` convert
  exactly; a float is read to 15 significant digits first. A value that does not fit the places
  without rounding (`1.005` at 2 places) stops the step, saying to round first, or with
  `on_invalid: set_null` is emptied, counted and marked.
- **`read_csv` reads `Decimal` columns**, named in `column_types` with `places`. Each value must be
  written as plain digits that fit the places, or the run stops naming it; a blank field is null.
  A column is never worked out as `Decimal` by itself.
- **`unpivot` takes `values_type: Decimal`** with `values_places`, converting every turned column
  exactly by `cast_columns`' rule. Without `values_type`, exact columns of different places, which
  were refused, are stacked at the larger places. An exact column beside a `Float64` one is refused,
  naming the float column.
- **`mean` and `median` of an exact `Decimal` column**, which were refused, are exact in `group_by`
  and `cross_tab`: the total divided by the count, or the middle value (the two middle values divided
  by 2), at 16 places or the column's own if more, the last rounded half away from zero. One that
  needs more than 38 digits stops the step. `std` and `quantile` still refuse an exact column, now
  saying why.
- **`join` pairs exact keys by value**: a `Decimal` key with a `Decimal` key of other places (1.50
  with 1.5000), and with a whole-number key (5.00 with 5). Both used to be refused, so no join that
  ran before gives a different answer. A `Decimal` key beside a `Float64` key is still refused, now
  naming the float column to convert. `make_groups` links two `Decimal` columns of different places
  by value too, at the larger places.
- **`write_excel` writes a `Decimal` column** when every value has at most 15 significant digits,
  the most a cell keeps; it used to refuse every `Decimal` column. A value with more refuses the
  column, naming the rows. A `formats`, `widths` or `wrap` rule for `type: Decimal` reaches every
  `Decimal` column, whatever its places.
- **The viewer shows an exact column's type in words**: `Decimal, 6 places` where it showed
  `Decimal(precision=38, scale=6)`, in each column heading and in a saved run's `ports.json`. Cells
  show the digits as held, including a tiny figure such as `0.0000001000`, which a drawn slice used
  to show as `1.000E-7`.
- **`union` stacks `Decimal` columns of different places** at the larger places (0.07 over 0.0725
  gives 0.0700 and 0.0725), where it used to refuse them. A `Decimal` column over a `Float64` or
  whole-number column is still refused; that refusal, and every other type refusal in `union`, now
  says to use `cast_columns` on one side.
- **`read_table` reads a Postgres `numeric` as an exact `Decimal` it controls.** A value a `Decimal`
  cannot hold (more than 38 digits, or `NaN`) stops the step naming the column, instead of an engine
  error. A `numeric` with no places stated still arrives at the places of its data; the notes now say
  so and recommend declaring them.

### Changed

- **`cast_columns` from an exact decimal to `Float64` lands on the nearest double.** It now goes
  through the value's digits. The figure changes only where the value's digits, counting every
  place the column holds, make a whole number larger than 2^53, or the column holds more than 22
  places: there the old conversion could miss by one step (457,200,000,000,000,000,000,000.00 gave
  4.5720000000000004e+23, and gives 4.572e+23 now). Any 2-place price under 90 trillion converts
  exactly as before. Nothing to do.
- **`unpivot` of exact columns gives 38 digits in all.** Turned exact columns that all held, say,
  `Decimal(18, 2)` came out as `Decimal(18, 2)`; they come out as `Decimal(38, 2)` now, with the same
  values. Nothing to do, unless a later step compared the type by its digits.
- **`group_by` and `cross_tab` give an exact total 38 digits in all.** A `sum`, `min`, `max`,
  `first` or `last` of a `Decimal(18, 2)` column gave `Decimal(18, 2)` (and on polars 1.38 a sum
  past 16 digits before the point came out wrong); it gives `Decimal(38, 2)` now, with the right
  value. Nothing to do.
- **An exact total past 38 digits stops the step.** A `sum` in `group_by`, or a `cross_tab` cell,
  of an exact column whose total needs more than 38 digits gave a wrong number without a word (and
  on polars 1.38 could crash the run later); it now stops, naming the group and the column. Convert
  the column to `Float64` for figures that large, or round it to fewer places first.
- **`read_table` types a Postgres `numeric(p,s)` from its declaration.** An empty table's column used
  to arrive as `Null`; it now arrives as `Decimal` with s places, as a full table's does. A column a
  driver handed over with fewer than 38 digits in all (`numeric(12,2)` through ADBC) now arrives with
  38, keeping its places. The figures are the same. If a later step compared that column's type
  with a `Null` one, look at it again.
- **`write_table` and `publish_tables` refuse a number a `numeric(p,s)` column would round.** Before,
  a `Float64` value with more places than the column holds (10.255 into `numeric(12,2)`) was rounded
  by Postgres without a word, and stored as it was by SQLite; a not-a-number went into Postgres as
  `NaN`; a whole number with too many digits was refused by Postgres in its own words and stored by
  SQLite. Now each stops the step before anything is written, naming the column, the declaration and
  the rows. A `Float64` that fits is written as its reading to 15 significant digits, which is what
  Postgres stored anyway for any figure of 15 digits or fewer. What to do: round the column in a
  formula step to the table's places (`round([amount], 2)`), or widen the table's column. Writes of
  `Float64` to `double precision`, `real` or `float` columns, and to a `numeric` with no places
  stated, are unchanged.
- **A `Decimal` column is refused by a `double precision`, `real` or `float` column**, where it used
  to be written as its nearest binary value. Convert it with `cast_columns` to `Float64` first. A
  `Decimal` column that fits now goes into a `numeric(p,s)` column on SQLite too, sent as its digits;
  before, SQLite refused it with a driver error.

## [0.3.0] - 2026-11-20

This release adds to the library and changes no result a pipeline gives today. It reads Excel's
older format, says which database a real run will write to, keeps saved runs private, and adds
date, number and text functions and more aggregates. Pipeline files written for 0.2 need no change.

### Added

- **`read_excel` reads `.xls`**, Excel's older format, as well as `.xlsx`. The same cells give the
  same rows, types and refusals in both, whichever day an `.xls` counts its dates from. A file named
  `.xls` that holds a web page or delimited text is refused, saying so. The viewer's **Show file**
  previews an `.xls` as a workbook.
- **The confirmation before a real run names the database.** For each table it lists the host, the
  port when one is stated, and the database name the connection points at now, in the viewer and in
  `dantron-etl run --write`. It never shows the user or the password. If the connection points at
  another database by the time you confirm, the run is refused.
- **`save_rows`** in `etl-project.yaml`, at the top or under one pipeline. With `save_rows: false` a
  run keeps its steps, timings, row counts, columns and types, and what it wrote, but no row of data.
  The viewer still shows the flow, the settings, the counts and the columns, and says where rows
  would be that they are not saved. A failed step's message is kept without the example rows it
  lists.
- **Date functions** `month`, `day`, `quarter` and `day_of_week` (Monday is 1), `add_days`,
  `month_start`, and `parse_date`, which reads text as a date in a stated format and stops the step
  on text that does not match, instead of leaving the date blank.
- **Number functions** `floor`, `ceil`, `sign`, `mod`, `least`, `greatest` and `nullif`. `mod`
  keeps the sign of the first number, as SQL does, and `mod` by zero stops the step, as `/` by zero
  does.
- **Text functions** `left`, `right`, `substr`, `split_part`, `pad_left` and `pad_right`. They count
  characters, not bytes. `pad_left` and `pad_right` never cut a longer value. `substr`'s length may
  be left out, the first argument in the language that may be.
- **Aggregates** for `group_by` and `cross_tab`: `mean`, `median`, `std` (the sample standard
  deviation), `quantile` with a `fraction` from 0 to 1, `last`, `last_non_null`, `any_true` and
  `all_true`. They skip nulls, and a group with no value gives null. The statistics refuse an exact
  `Decimal` column; cast it to `Float64` first.

### Security

- **Saved runs are readable by their owner only** on Linux and macOS. Each folder a run makes is
  `0700` and each file in it `0600`, whatever the umask. Folders you made, older runs and the files
  your output steps write are left as they are. On Windows, and on a drive that refuses the change,
  the run is saved without it.

### Changed

- The package's **Documentation** and **Changelog** links, and the README's Postgres install line,
  point at `main`, the latest release, instead of the working branch.
- **For tool authors: `refuse_division_by_zero` is now `refuse_rows_without_an_answer`**, in
  `dantron_etl.authoring`, since it stops a step on any row an expression has no answer for. There
  is no alias; a tool of your own that called the old name must call the new one. Pipeline files
  are not affected.

### Fixed

- **`add_months` stops the step on a count that takes a date past what the engine can hold**, and
  names the count and the rows, instead of crashing the engine. Dates run from the year -262143 to
  262142; a timestamp kept to the nanosecond runs from 1677 to 2262, and moved past that it used to
  give a wrong date without a word. A count written in the file that no date from the year 1 to
  9999 can take (beyond 3,145,703 forward or 3,265,715 back) is refused when the file loads. Every
  other count gives the date it gave before.
- **`add_days` stops the step on a timestamp kept to the nanosecond moved past 1677 to 2262**,
  where it gave a wrong date without a word, and refuses a count written in the file beyond 106,751
  days when the file loads.
- **`mod` of the most negative whole number** (-9223372036854775808, or a 32-bit column's
  -2147483648) now gives the right remainder, with the sign of the first number: `mod` of it by 3 is
  -2, where it gave -1.
- **A null with no type gives a null answer** in `year`, `month`, `day`, `quarter`, `day_of_week`,
  `add_months`, `add_days` and `abs`, instead of an engine error. That is the literal `null`, or a
  column whose every value is missing and which was read with no type. `year` and the other parts
  of a date give a null whole number (Int32); `abs`, `add_months` and `add_days` give a null with no
  type, as `round` already did, since their answer takes the type of what went in.

## [0.2.0] - 2026-11-20

This release makes the library easier to start with and to read. It adds a command and a viewer
you can open from any project, renames the tools and settings whose names said less than they
meant, and changes a few results that were quietly wrong. Pipeline files written for 0.1 are
refused until they are migrated; one command does it. See [docs/upgrading.md](docs/upgrading.md).

### Added

- The library is released under the Apache License 2.0 (`LICENSE`, `NOTICE`).
- **The `dantron-etl` command.** `new` makes a project with a working example. `view` opens the
  viewer in your browser. `run` runs a preview, or a real run with `--write`. `check` checks a file
  without running it. `tools`, `help <tool>` and `functions` describe the library in the terminal.
  `catalog` writes the tool catalog. `migrate` rewrites old pipeline files.
- **The viewer is part of the package.** It lists a project's pipelines with their last run, runs a
  preview from the page, and asks you to confirm before a real run, listing every file and table it
  will write. It listens on this computer only. A failed run is kept beside the last good one.
- **A project file**, `etl-project.yaml`, saying where pipelines and runs live, and each pipeline's
  rules and the column the viewer follows. It is optional.
- **Launchers** to double-click on Windows and macOS, and a VS Code task, in every new project.
- **Expression functions** `months_between`, `days_between`, `make_date` and `month_end`, so a
  pipeline no longer turns dates into numbers to subtract them.
- **`first_non_null`**, an aggregate giving the first value that is there.
- **`allow_empty`** on `write_csv` and `write_excel`, and **`when_incoming_is_null`** on a
  `publish_tables` merge.
- **`base_dir`** on `run_pipeline`, so relative paths can be read from a folder you name.
- **Everything needed to load and run a pipeline is importable from `dantron_etl`** itself.
- **Documentation for people** under `docs/`: a first-pipeline walk-through, concepts, the pipeline
  file, expressions, a page per tool, a cookbook, checking your figures, the command and upgrading.
- `py.typed`, so type checkers read the library's annotations.
- **Show file, in the viewer.** Each setting that names a file (`path` on the read and write tools,
  `template` on `write_excel`) has a **Show file** button. It shows the full path the run uses,
  whether the file is there, its size and when it last changed, and offers **Preview** (the first
  rows of a CSV or a sheet of a workbook) and, where the viewer can reach a file manager, **Open
  location** (Explorer, Finder, Explorer from WSL, or a Linux desktop's file manager). The real-run
  confirmation shows the same for each file it is about to write. A tool whose setting is marked as
  a file path gets this without any change to the viewer.

### Changed

- **Renamed tools, settings, ports, values and one function** so their names say what they do: for
  example `join`'s `matched` port is now `joined`, `write_workbook` is now `write_excel`,
  `connected_components` is now `make_groups`, a pipeline's `groups` are now `containers`, and
  `strip()` is now `trim()`. The full list is in [docs/upgrading.md](docs/upgrading.md). An old
  name is refused with the new one, and `dantron-etl migrate --write` rewrites the files.
- **`round` rounds a half away from zero**, as a spreadsheet does, on the number as written:
  `round(1.005, 2)` is `1.01`. `format_number` agrees with it.
- **Dividing by zero stops the step** and names the rows, instead of giving infinity.
- **`count_distinct` (was `n_unique`) skips blanks**, as SQL's `COUNT(DISTINCT)` does.
- **A decimal cast to a whole number stops** unless every value is already whole, instead of
  cutting off the fraction.
- **A timestamp with a time zone needs `in_zone`** to become a date, in `cast_columns`, `date_diff`
  and `expand_date_range`.
- **`write_csv` and `write_excel` stop on an empty result** unless the step allows it.
- **`write_excel` refuses not-a-number and infinity** instead of writing a blank cell, and builds
  each workbook from its template or from nothing, never from the file already at its path.
- **A pipeline file with a key the loader does not know is refused**, with the nearest name.
- **Many more mistakes are refused when the file loads**, before any data is read: a CSV writer's
  missing folder, join keys whose types cannot be compared, expressions that can never be true.
- `date_diff`'s `counting` is now required, and time units are spelled in the singular.
- The test suite runs in about a minute and a half instead of four.
- The version is written in one place, and `dantron-etl --version` prints it.
- **Python 3.11 and 3.12 are supported**, not only 3.13. CI runs the tests on all three.
- An engine error inside a refusal no longer carries the engine's own note about its internal
  expression, which newer polars versions add.

### Removed

- The unused SQL output of the expression compiler. Expressions compile to polars only.
- The old single-sheet `write_excel` (the one with a `sheet` setting). Use `write_excel` with one
  entry under `sheets`.
- The old module paths `dantron_etl.pipeline` and `dantron_etl.tool`, and the viewer scripts that
  lived outside the package.

### Security

- The viewer listens on `127.0.0.1` by default, refuses requests addressed to any other host, and
  serves only the files a run refers to.
- The source package lists what goes into it, so a local `.env`, key or data file cannot ship.

## [0.1.0] - 2026-11-10

The first version: the tools, the pipeline file, the expression language and the run trace.

[Unreleased]: https://github.com/DanTron-Inc/dantron-etl-tools/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DanTron-Inc/dantron-etl-tools/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DanTron-Inc/dantron-etl-tools/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DanTron-Inc/dantron-etl-tools/releases/tag/v0.1.0
