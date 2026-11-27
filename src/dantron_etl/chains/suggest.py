"""A first chain file, worked out from what a set of pipelines read and write.

Each pipeline is set to run after every pipeline that writes a file or table it reads, and after
nothing else. The text it gives says, beside each ``after``, which files and tables pass along it.
It is a starting point for a person to read: an order the files do not show is not in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..errors import ToolError, name_choices
from ..expr import ExpressionError
from ..viewer.project import Project, load_entry
from .chain_file import ChainError, Member, run_order
from .handoffs import Touch, touches

__all__ = ["suggest_chain"]


def suggest_chain(
    project: Project, name: str, pipelines: Sequence[str] = (), parameters: Mapping[str, Any] | None = None
) -> str:
    """The text of a chain file that runs ``pipelines`` (every pipeline in the project when empty).

    Each pipeline is loaded with its own rules. ``parameters`` become the chain's parameters, and
    each is handed to the pipelines that use it, as a run of the chain would. Refused when a
    pipeline is unknown or does not load, when a parameter is used by none of them, and when the
    pipelines' reads and writes go round in a loop, which no order can satisfy.
    """
    import yaml

    names = list(dict.fromkeys(pipelines)) or list(project.pipelines)
    if not names:
        raise ChainError(f"The project at {project.root} has no pipelines to put in a chain.")
    values = dict(parameters or {})
    for key in values:
        if not any(key in project.pipelines[one].rules_needed for one in names if one in project.pipelines):
            raise ChainError(f"No pipeline given uses the parameter {key}: none of them writes ${{rules.{key}}}.")
    reads: dict[str, tuple[Touch, ...]] = {}
    written: dict[str, tuple[Touch, ...]] = {}
    for one in names:
        entry = project.pipelines.get(one)
        if entry is None:
            raise ChainError(f"There is no pipeline {one!r} in the project; {name_choices(one, project.pipelines, 'pipelines')}")
        try:
            spec = load_entry(project, entry, {k: v for k, v in values.items() if k in entry.rules_needed})
        except (ToolError, ExpressionError, yaml.YAMLError, OSError) as refusal:
            raise ChainError(f"{one} does not load, so what it reads cannot be worked out: {refusal}") from refusal
        reads[one], written[one], unknown = touches(spec, project.root)
        if unknown:
            raise ChainError(f"What {one} reads or writes at step {', '.join(unknown)} cannot be worked out.")

    passed: dict[str, dict[str, list[str]]] = {one: {} for one in names}
    outside: dict[str, list[str]] = {one: [] for one in names}
    for reader in names:
        for read in reads[reader]:
            writers = [w for w in names if w != reader and any(t.target.matches(read.target) for t in written[w])]
            for writer in writers:
                passed[reader].setdefault(writer, [])
                if read.target.shown not in passed[reader][writer]:
                    passed[reader][writer].append(read.target.shown)
            if not writers and not any(t.target.matches(read.target) for t in written[reader]):
                if read.target.shown not in outside[reader]:
                    outside[reader].append(read.target.shown)

    members = [Member(one, tuple(passed[one])) for one in names]
    order = run_order(members, "the pipelines given, by what they read and write")
    lines = [
        "# A first chain, worked out by 'chain suggest' from what each pipeline reads and writes.",
        "# Each pipeline runs after the pipelines that write a file or table it reads. Check it before",
        "# you run it: an order the files and tables do not show is not in it.",
        f"title: {_plain(name)}",
        "",
    ]
    if values:
        lines.append("parameters:")
        lines += [f"  {line}" for line in yaml.safe_dump(values, sort_keys=False).splitlines()]
        lines.append("")
    shared = {rule: users for rule, users in _shared_rules(project, names).items() if rule not in values}
    if shared:
        lines.append("# These rules are used by more than one pipeline. Each could be a parameter of the chain:")
        lines += [f"#   {rule}: used by {', '.join(users)}" for rule, users in shared.items()]
        lines.append("")
    lines.append("pipelines:")
    for one in order:
        lines.append(f"  - name: {one}")
        if passed[one]:
            lines.append(f"    after: [{', '.join(passed[one])}]")
            for writer, targets in passed[one].items():
                lines.append(f"    # from {writer}: {', '.join(targets)}")
        if outside[one]:
            lines.append(f"    # from outside the chain: {', '.join(outside[one])}")
    return "\n".join(lines) + "\n"


def _shared_rules(project: Project, names: Sequence[str]) -> dict[str, list[str]]:
    users: dict[str, list[str]] = {}
    for one in names:
        for rule in project.pipelines[one].rules_needed:
            users.setdefault(rule, []).append(one)
    return {rule: found for rule, found in sorted(users.items()) if len(found) > 1}


def _plain(text: Any) -> str:
    """``text`` as a YAML value that reads back as the same text."""
    import yaml

    return yaml.safe_dump(str(text), default_style=None).splitlines()[0].removesuffix("...").strip()
