# Changelog

What changed in each release, for people who use the library. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor version (0.2, 0.3) may rename
things; each such release says how to upgrade.

## [Unreleased]

### Added

- **Date functions** `month`, `day`, `quarter` and `day_of_week` (Monday is 1), `add_days`,
  `month_start`, and `parse_date`, which reads text as a date in a stated format and stops the step
  on text that does not match, instead of leaving the date blank.

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

[Unreleased]: https://github.com/DanTron-Inc/dantron-etl-tools/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/DanTron-Inc/dantron-etl-tools/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DanTron-Inc/dantron-etl-tools/releases/tag/v0.1.0
