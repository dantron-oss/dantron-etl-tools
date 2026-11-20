## What this changes

One or two plain sentences, for someone who uses the library rather than works on it.

## Could it change a figure?

Say whether a pipeline that runs today could give a different result, or be refused, after this
change. If so, say which tools or settings, and what is in CHANGELOG.md about it.

## Checks

- [ ] `pytest tests/ -q` and `ruff check src tests docs/_build` pass
- [ ] Each new rule is tested on both sides (the other port, the empty table, the blank value)
- [ ] Generated pages rewritten if a tool, function, key, rename or command changed (`python docs/_build/generate.py`)
- [ ] A renamed name is in `renames.py` and the vocabulary snapshot
- [ ] New prose is plain: what it does first, short sentences
- [ ] Nothing names a client, here or in the commit messages
