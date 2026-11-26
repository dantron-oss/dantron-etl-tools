# Contributing

Thank you for helping. This page says how to set up, how to run the tests, what to check before you
add a tool or a function, and what a pull request is held to.

For a change bigger than a fix, open an issue first and describe what you want to do. It is much
easier to agree on the shape of a change before the code exists.

## Set up

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Development uses 3.13, the
version in `.python-version`, and CI also tests 3.11 and 3.12.

```bash
git clone https://github.com/DanTron-Inc/dantron-etl-tools
cd dantron-etl-tools
uv sync --extra dev --extra postgres
```

This installs the library in editable mode, with the exact versions in `uv.lock`, into `.venv`.
On Windows, use `.venv\Scripts\` wherever this page says `.venv/bin/`.

`make` lists shortcuts for everything below. Each runs one command, so on Windows you can type the
command yourself.

## Run the tests

```bash
.venv/bin/pytest tests/ -q          # about 4,900 tests, about a minute and a half
.venv/bin/ruff check src tests docs/_build
```

The tests are in folders by area: `core/` (loading, running, the trace, the database plumbing),
`expr/` (the expression language), `tools/` (one file per tool, named after it), `viewer/` (the
page, projects, runs, the server), `package/` (what ships: the boundary checks, the source package,
the documentation examples, the vocabulary snapshot) and `flows/` (whole pipelines end to end). Test
file names must be unique across folders.

`-m "not slow"` leaves out the slowest tests while you work.

### Both ends of the polars range

The suite must pass on the lowest polars the package allows (the `polars>=` bound in
`pyproject.toml`) and on the version in `uv.lock`. Projects run either one, and the two give
different answers in a few places, most of all for exact decimals. A test that passes on one and
fails on the other is a defect: the library must give one answer on both. CI runs both.

`.venv` holds the locked version, so the command above tests that end. To test the lowest end, make
a second environment beside it and put polars at the bound there:

```bash
UV_PROJECT_ENVIRONMENT=.venv-lowest uv sync --extra dev --extra postgres
uv pip install --python .venv-lowest --resolution lowest --upgrade-package polars "polars>=1.38,<2"
.venv-lowest/bin/pytest tests/ -q
```

Write the requirement as `pyproject.toml` has it today. Git ignores `.venv-lowest`. A later `uv sync`
into it puts the locked polars back, so run the `uv pip install` line again after one.

### The Postgres tests

About 50 tests need a Postgres server. They skip without one. To run them, start the throwaway
server in `compose.yaml` (it needs Docker) and say where it is:

```bash
docker compose up -d --wait
export DANTRON_TEST_POSTGRES=postgresql://etl_test:etl_test@localhost:5434/etl_test
.venv/bin/pytest tests/ -q
docker compose down
```

In PowerShell: `$env:DANTRON_TEST_POSTGRES = "postgresql://etl_test:etl_test@localhost:5434/etl_test"`.
Never point it at a server holding real data.

### Words that identify your clients

If you use this library for clients, their names and the names of their systems must never appear
in this repository: not in code, tests, docs, commit messages, branch names or tags. A test checks
every tracked file for such words, but the list of words cannot live in the repository, because it
would name the clients itself.

So each maintainer keeps the list locally, in a file called `.forbidden-terms` at the top of the
checkout. Git ignores it. Write one word per line; `#` starts a comment; a word matches anywhere,
ignoring case.

```text
# words that identify a client or its systems
acme
acmeledger
```

The environment variable `ETL_FORBIDDEN_TERMS` (comma-separated) adds more; CI reads it from a
repository secret of the same name. With neither, the check is skipped and says so. A word that
could be a Python package name is also refused as an import anywhere in the library or its tests.

### The generated pages

Some pages under `docs/` are generated from the library: the tool pages, the function table, the
key tables, the rename table and the command help. After changing a tool, a function, a loader key,
a rename or the command, run:

```bash
.venv/bin/python docs/_build/generate.py
```

and commit what it writes. A test fails while they are out of date.

## Before you add a tool

**A new tool is a last resort.** Most capabilities that look missing are two existing tools in a
row: a type change then a calculation, or a reshape then a fill. Before proposing a tool, answer
these in order, in the issue:

1. Can one tool do it? Search the [tool pages](docs/tools/README.md) by what you want to end up
   with, not by the verb you first thought of.
2. Can two tools in a row do it? Three?
3. Is it one new column worked out from the others? Then it is an expression **function**, not a
   tool.
4. Only then: is it a new way of reshaping the table?

Describe the shape of the change, not the tool you imagined: what is in the table, what must come
out, the tools you tried and where they stopped working, and a small example of rows in and rows
out.

**Every tool must be generic.** It must be useful on another project's data. A tool that exists for
one report is a bug.

### Adding a tool

A tool is one file in `src/dantron_etl/tools/`, registering itself at its foot. `tools/filter.py`
is the smallest complete example. A tool imports everything it needs from `dantron_etl.authoring`.
It needs:

1. A `name`, a one-line `description`, and a `category`.
2. Named `input_ports` and `output_ports`.
3. Every setting typed, described, and either required or given a default.
4. `help()`: a description, at least one worked example, notes stating its rulings (what it does
   with blanks, what it refuses, which SQL operation it follows), and help for each setting.
5. `validate_config` for rules across settings, and `validate_inputs` for everything that can be
   checked from the columns alone.
6. Tests, including the mirror of each rule (see below).
7. A row in `src/dantron_etl/MANUAL.md` section 3.1, the count updated there and in `README.md`,
   and the generated pages rewritten.

### Adding a function

A function is one entry in the function table, `DEFINITIONS` in `src/dantron_etl/expr/functions.py`:
the name, the arguments in order, the type of the result, the code that builds the column, a
one-line summary and an example. If a function has no answer for some rows, such as text it cannot
read, say which in the entry's `no_answer`: every tool then stops the step on those rows instead of
writing a null. Add its tests, list it in `MANUAL.md` section 4.2, and rewrite the generated pages. A function never adds syntax to the language: no new operators, no variables, no
functions defined in a pipeline.

### Renaming something

A tool, setting, port, value or function name that a pipeline file uses is a promise to everyone
with a pipeline. To rename one:

1. Add the old and new name to `src/dantron_etl/renames.py`. The loader then refuses the old name
   with the new one, and `dantron-etl migrate` rewrites old files.
2. Update the vocabulary snapshot, `tests/fixtures/vocabulary.json`, in the same commit. Its test
   fails on any change to a name a pipeline can write, so a rename is always on purpose.
3. Say so in `CHANGELOG.md`.

## What a pull request is held to

- **Every worked example runs.** The conformance check runs each tool's examples and compares the
  output. A tool that needs a database writes its example against an in-memory SQLite.
- **Refuse early, by name.** A setting that cannot work on a table's columns is refused whether the
  table has rows or not. The message names the step, the setting, the column, what it held and what
  was asked of it. Nothing fails quietly: a step that matches nothing while reporting success is the
  worst outcome this library can have.
- **Each tool follows the SQL operation it is named after**, and its notes say which. That decides
  what it does with blanks.
- **Test the mirror case.** For each rule, test the other side too: the other port, the other order,
  the empty table, the blank value. Then break the code on purpose and check a test fails.
- **A promise in the docs has a test.** If a description says the tool does something, a test fails
  when it stops doing it.
- **No input is changed in place**, and no row-by-row Python inside a tool.
- **Only polars and the standard library are imported at module level.** Anything else is imported
  inside the function that needs it.
- **A pipeline file holds no secrets.** A setting names an environment variable; it never holds a
  password.
- **No `eval`, `exec` or attribute lookups on text from a pipeline file.**
- **Plain writing.** Docstrings, descriptions, notes and commit messages are read by people who are
  not developers. Say what a thing does first, in one short sentence. Short sentences, one idea
  each, everyday words. Long reasoning belongs in a note under `docs/design/`.
- **Client-neutral.** No client's name or vocabulary anywhere, including commit messages and branch
  names.
- **Lint passes and every test passes**, on Linux and Windows.

## Commits

Commit titles are plain sentences saying what changed for the reader, such as "Publish a set of
related tables, all of them or none of them". The body says why. A change that alters what a
pipeline computes says so plainly.

Work happens on a `feature/<name>` branch and merges into `develop`.

## Releasing a version

Work merges into `develop`. `main` holds the latest released version and moves only when a version
is released, because projects install `@main` to get the latest release. Each release is also a tag
on `main`, named `v` and the version, such as `v0.2.0`, so a project can stay on one version. The
code under a tag never changes once it is pushed; a fix goes out as a new version.

1. Merge everything for the release into `develop`, and check the suite passes.
2. Set `__version__` in `src/dantron_etl/__init__.py` to the new version. `pyproject.toml` reads it
   from there. Before 1.0, raise the middle number (0.2 to 0.3) when pipeline files or results
   change, and the last number (0.2.0 to 0.2.1) for a fix that changes neither.
3. In `CHANGELOG.md`, rename the `[Unreleased]` heading to the version and today's date, start a new
   empty `[Unreleased]` above it, and update the comparison links at the bottom.
4. Commit, merge `develop` into `main`, and push both.
5. Tag `main` and push the tag:

   ```bash
   git tag -a v0.3.0 -m "Version 0.3.0"
   git push origin v0.3.0
   ```

## Licence

The library is under the Apache License 2.0. By contributing you agree that your work is offered
under the same licence.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
