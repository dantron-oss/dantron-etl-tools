# Paths: finding the project root

Notes on `paths.py`. Every caller that reads or writes a project's files (the viewer, the command,
a client's own code) hangs those places off one folder: the root of the caller's project.
`project_root` is the only place that question is answered. It has one dangerous wrong answer, and
answering it in several places means writing that wrong answer several times.

## The wrong answer: the library's own file

**Rejected.** Finding the root by walking up from the library's own file:

```python
ROOT = Path(__file__).resolve().parents[2]
```

This is right while the package sits in the `src/` folder of its own repository. It is silently
wrong once the same line ships in an installed package:

```
/client/.venv/lib/python3.13/site-packages/<pkg>/paths.py  ->  parents[2]
/client/.venv/lib/python3.13
```

Nothing raises. That folder exists, can be read, and is not the project, so every path joined onto
it is a real path to the wrong place. The project this library came from paid for that twice in
one day: a workbook built from another checkout's pipeline file, and a run that read a 16-step flow
where a 672-step one was meant. Both looked ordinary going past.

So the design aims at refusing, not at resolving. There is no module-level root constant for a wrong
answer to be baked into at import time.

## The caller says where it is

**Decided.** `project_root(start)` takes the caller's location, normally its own `__file__`, and
searches upward from there. The argument has no default.

**Why.** A default would have to be this module's location or the working directory, and both are
rejected here. The cost is one explicit argument at each call site. That cost was chosen on purpose,
against a kind of defect that had already cost two rebuilds.

The search is right in a repository, in a linked worktree, for a loose script and for an installed
module. When no marker is found there is nothing plausible to return, so the outcome is a named
exception, `ProjectRootNotFound`, instead of a path.

**Rejected: the environment variable, else the working directory.** This suits a command a person
runs from inside a project, and it is wrong for a library. The working directory of a long-running
process (a daemon, a notebook kernel, a scheduler) depends on how it was launched, and it can change
while the process runs. It also fails the harmful way: it answers plausibly instead of refusing.

This design is longer than that one, and someone will want the shorter one back. This note is why
not.

## The environment variable comes first, and is checked

**Decided.** `ETL_PROJECT_ROOT` names the root outright. It is read before the search, and the folder
it names must itself be a project.

**Why it exists.** It is how a process is told which checkout it belongs to when several are on disk
at once, for example in a container or a CI job.

**Why it comes first.** The case it exists for is the one where the search would succeed with the
wrong answer.

**Why it is checked.** A variable pointing at a folder that is not a project is a typo in a
deployment. Trusting it because someone set it on purpose is how a stale checkout goes unnoticed for
a day. Every refusal names the variable, because setting it is the one move a reader can make.

## What marks a project

**Decided.** A folder is a project root if it holds `pyproject.toml`, `.git` or `etl-project.yaml`,
as a file or as a folder.

- `pyproject.toml` alone is wrong somewhere real: a source checkout that has not been packaged has
  none.
- `.git` alone is wrong too: an exported or copied tree has none.
- A folder holding the project settings file, `etl-project.yaml`, is a project even with neither.
- Markers are matched as files or folders because a linked git worktree's `.git` is a file holding a
  `gitdir:` line. A check using `is_dir()` would refuse every worktree.

`ProjectRootNotFound` descends from `RuntimeError`, so a caller that catches everything still
catches it, and has its own name so a caller can handle only this. Its name breaks the usual
`...Error` suffix rule because the sentence reads better.
