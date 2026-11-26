# Changelog

What changed in each release, for people who use the library. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version (0.2, 0.3) may rename
things; each such release says how to upgrade.

## [Unreleased]

### Added

- **`Decimal` is a type name**, for exact decimals, with `places` (0 to 38) stated beside it. It is
  listed in `COLUMN_TYPES`. No tool takes it yet: `cast_columns`, `read_csv`, `unpivot` and
  `write_excel`'s format, width and wrap rules refuse it, now with a message that says so.
- **`join` pairs exact keys by value**: a `Decimal` key with a `Decimal` key of other places (1.50
  with 1.5000), and with a whole-number key (5.00 with 5). Both used to be refused, so no join that
  ran before gives a different answer. A `Decimal` key beside a `Float64` key is still refused, now
  naming the float column to convert. `make_groups` links two `Decimal` columns of different places
  by value too, at the larger places.
- **`union` stacks `Decimal` columns of different places** at the larger places (0.07 over 0.0725
  gives 0.0700 and 0.0725), where it used to refuse them. A `Decimal` column over a `Float64` or
  whole-number column is still refused; that refusal, and every other type refusal in `union`, now
  says to use `cast_columns` on one side.

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
