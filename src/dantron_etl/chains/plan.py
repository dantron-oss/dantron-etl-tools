"""A chain worked out before it runs: its pipelines loaded, its order checked against what they pass on.

:func:`load_chain` reads the chain file, loads every pipeline in it with the chain's parameter
values, and refuses the chain when the order it states disagrees with the files and tables the
pipelines read and write. Nothing runs and nothing is written.

Why: docs/design/chains.md
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..errors import ToolError, name_choices
from ..expr import ExpressionError
from ..spec import Pipeline
from ..viewer.project import PipelineEntry, Project, combine_rules, find_project, own_rules, read_entry
from ..viewer.runs import Write, writes, writes_digest
from .chain_file import ChainError, ChainFile, read_chain_file, run_order, value_for
from .handoffs import Target, Touch, touches

__all__ = ["ChainPlan", "Handoff", "MemberPlan", "load_chain", "open_project"]


@dataclass(frozen=True)
class Handoff:
    """One file or table one pipeline in the chain writes and a later one reads."""

    target: Target
    writer: str
    writer_step: str
    reader: str
    reader_step: str


@dataclass(frozen=True)
class MemberPlan:
    """One pipeline of a chain, loaded with the values the chain gives it."""

    name: str
    title: str
    after: tuple[str, ...]
    why: str | None
    steps: int
    #: The chain's parameter values this pipeline uses, by name.
    given: Mapping[str, Any]
    #: The pipeline's own value of each parameter the chain replaces, by name.
    replaced: Mapping[str, Any]
    reads: tuple[Touch, ...]
    writes: tuple[Touch, ...]
    #: What a real run writes, as the viewer and the run command list it, and its digest.
    write_list: tuple[Write, ...]
    digest: str


@dataclass(frozen=True)
class ChainPlan:
    """A checked chain: its pipelines in run order, their hand-offs, and what it reads and writes."""

    project: Project
    name: str
    title: str
    path: Path
    max_parallel: int
    #: Every parameter's value for this run: the default, or the value given.
    values: Mapping[str, Any]
    #: The values given for this run, as given. Running the chain again with them gives this plan.
    asked: Mapping[str, Any]
    members: Mapping[str, MemberPlan]
    order: tuple[str, ...]
    handoffs: tuple[Handoff, ...]
    #: Files and tables a pipeline reads that no pipeline in the chain writes, with who reads each.
    outside: tuple[tuple[Target, tuple[str, ...]], ...]
    #: Files and tables a pipeline writes that no pipeline in the chain reads, with who writes each.
    results: tuple[tuple[Target, str], ...]
    digest: str

    def after(self, name: str) -> tuple[str, ...]:
        """The pipelines ``name`` runs after, as the chain file lists them."""
        return self.members[name].after

    def before_all(self, name: str) -> frozenset[str]:
        """Every pipeline that must finish before ``name`` starts, however far back."""
        return _before_all(self.members, name)

    def after_all(self, name: str) -> frozenset[str]:
        """Every pipeline that waits for ``name``, however far on."""
        return frozenset(other for other in self.order if name in self.before_all(other))

    def handoffs_between(self, writer: str, reader: str) -> tuple[Handoff, ...]:
        return tuple(one for one in self.handoffs if one.writer == writer and one.reader == reader)

    def select(
        self, start_from: str | Collection[str] | None = None, until: str | Collection[str] | None = None
    ) -> tuple[str, ...]:
        """The pipelines a run covers, in run order.

        ``start_from`` runs those pipelines and every one after them. ``until`` runs those and every
        one before them. Given both, the run covers what lies between. Neither: the whole chain.
        """
        chosen = set(self.order)
        if start_from is not None:
            names = self._names(start_from, "--from")
            chosen &= set(names).union(*(self.after_all(name) for name in names))
        if until is not None:
            names = self._names(until, "--until")
            chosen &= set(names).union(*(self.before_all(name) for name in names))
        if not chosen:
            raise ChainError(
                f"No pipeline of chain '{self.name}' comes after {start_from} and before {until}, so there is "
                f"nothing to run."
            )
        return tuple(name for name in self.order if name in chosen)

    def _names(self, asked: str | Collection[str], option: str) -> tuple[str, ...]:
        names = (asked,) if isinstance(asked, str) else tuple(asked)
        if not names:
            raise ChainError(f"{option} names no pipeline. Name one of the chain's pipelines.")
        for name in names:
            if name not in self.members:
                raise ChainError(
                    f"{option} names {name!r}, which is not in chain '{self.name}'; "
                    f"{name_choices(name, self.order, 'its pipelines')}"
                )
        return names


def open_project(where: Project | Path | str) -> Project:
    """The project itself, or the project holding the folder or file ``where``."""
    return where if isinstance(where, Project) else find_project(Path(where))


def load_chain(
    where: Project | Path | str, name: str, parameters: Mapping[str, Any] | None = None
) -> ChainPlan:
    """Read a chain, load its pipelines with its parameter values, and check its order.

    ``where`` is the project, or any folder or file inside it. ``parameters`` changes parameter
    values for this run; a value given as text is read as the same kind as the default. Refused with
    a :class:`ChainError` naming what to change: an unknown chain, pipeline or parameter, a pipeline
    that does not load, a parameter no pipeline uses, and every way the order disagrees with what the
    pipelines read and write. This may run project code, the way loading one pipeline does.
    """
    project = open_project(where)
    if name not in project.chains:
        raise ChainError(f"There is no chain {name!r} in the project at {project.root}; {name_choices(name, project.chains, 'chains')}")
    chain = read_chain_file(project.chains[name])
    shown = f"chain '{name}'"
    asked = dict(parameters or {})
    for key in asked:
        if key not in chain.parameters:
            known = ", ".join(chain.parameters) or "none"
            raise ChainError(f"A value was given for {key!r}, which is not a parameter of {shown}. Its parameters: {known}.")
    values = {
        key: value_for(key, default, asked[key], shown) if key in asked else default
        for key, default in chain.parameters.items()
    }
    entries = _entries(project, chain)
    unused = [key for key in values if not any(key in entry.rules_needed for entry in entries.values())]
    if unused:
        raise ChainError(
            f"No pipeline in {shown} uses the parameter{'s' if len(unused) > 1 else ''} "
            f"{', '.join(unused)}. A pipeline uses one by writing ${{rules.{unused[0]}}}. Remove it from "
            f"'parameters' or check its spelling."
        )
    order = run_order(chain.members, shown)
    members = {one: _member(project, chain, entries[one], values) for one in order}
    handoffs, outside, results = _check(chain, members)
    return ChainPlan(
        project=project,
        name=name,
        title=chain.title,
        path=chain.path,
        max_parallel=chain.max_parallel,
        values=MappingProxyType(values),
        asked=MappingProxyType(asked),
        members=MappingProxyType(members),
        order=tuple(members),
        handoffs=handoffs,
        outside=outside,
        results=results,
        digest=_digest(members),
    )


def _entries(project: Project, chain: ChainFile) -> dict[str, PipelineEntry]:
    found: dict[str, PipelineEntry] = {}
    for member in chain.members:
        entry = project.pipelines.get(member.name)
        if entry is None:
            raise ChainError(
                f"Chain '{chain.name}' runs {member.name!r}, which is not a pipeline in the project; "
                f"{name_choices(member.name, project.pipelines, 'pipelines')}"
            )
        if entry.problem:
            raise ChainError(f"Chain '{chain.name}' runs {member.name}, which cannot be read: {entry.problem}")
        found[member.name] = entry
    return found


def _member(project: Project, chain: ChainFile, entry: PipelineEntry, values: Mapping[str, Any]) -> MemberPlan:
    import yaml

    member = chain.member(entry.name)
    given = {key: value for key, value in values.items() if key in entry.rules_needed}
    try:
        own = own_rules(project, entry)
        spec: Pipeline = read_entry(project, entry, combine_rules(entry, own, given))
    except (ToolError, ExpressionError, yaml.YAMLError, OSError) as refusal:
        raise ChainError(f"Chain '{chain.name}' runs {entry.name}, which does not load: {refusal}") from refusal
    reads, written, unknown = touches(spec, project.root)
    if unknown:
        raise ChainError(
            f"Chain '{chain.name}' cannot tell what {entry.name} reads or writes at step "
            f"{', '.join(unknown)}, so it cannot check the order. This is a gap in the library: report it."
        )
    found = writes(spec, project.root)
    return MemberPlan(
        name=entry.name,
        title=entry.title,
        after=member.after,
        why=member.why,
        steps=len(spec.steps),
        given=MappingProxyType(given),
        replaced=MappingProxyType({key: own[key] for key in given if key in own and own[key] != given[key]}),
        reads=reads,
        writes=written,
        write_list=found,
        digest=writes_digest(found),
    )


def _check(
    chain: ChainFile, members: Mapping[str, MemberPlan]
) -> tuple[tuple[Handoff, ...], tuple[tuple[Target, tuple[str, ...]], ...], tuple[tuple[Target, str], ...]]:
    """Every hand-off, or a refusal listing each way the order disagrees with the reads and writes."""
    shown = f"chain '{chain.name}'"
    problems: list[str] = []
    before = {name: _before_all(members, name) for name in members}

    written: list[tuple[str, Touch]] = [(name, touch) for name, plan in members.items() for touch in plan.writes]
    for at, (name, touch) in enumerate(written):
        for other, other_touch in written[at + 1 :]:
            if other != name and touch.target.matches(other_touch.target):
                problems.append(
                    f"{name} (step {touch.step}) and {other} (step {other_touch.step}) both write "
                    f"{touch.target.shown}. Only one pipeline in a chain may write a file or table; otherwise "
                    f"which version is kept depends on which finishes last."
                )

    handoffs: list[Handoff] = []
    outside: dict[Target, list[str]] = {}
    for reader, plan in members.items():
        for read in plan.reads:
            writers = [(name, touch) for name, touch in written if name != reader and touch.target.matches(read.target)]
            if not writers and not any(touch.target.matches(read.target) for touch in plan.writes):
                outside.setdefault(read.target, [])
                if reader not in outside[read.target]:
                    outside[read.target].append(reader)
            for writer, touch in writers:
                if writer in before[reader]:
                    handoffs.append(Handoff(read.target, writer, touch.step, reader, read.step))
                elif reader in before[writer]:
                    problems.append(
                        f"{reader} reads {read.target.shown} (step {read.step}), which {writer} writes, but "
                        f"{writer} runs after {reader}, so {reader} would read what an earlier run left there. "
                        f"Reverse the order: {reader} must run after {writer}."
                    )
                else:
                    problems.append(
                        f"{reader} reads {read.target.shown} (step {read.step}), which {writer} writes, but "
                        f"{reader} does not run after {writer}. They could run at the same time, and {reader} "
                        f"would read what an earlier run left there. Add {writer} to {reader}'s 'after'."
                    )
    if problems:
        lines = "\n".join(f"- {problem}" for problem in dict.fromkeys(problems))
        raise ChainError(
            f"The order of {shown} does not match what its pipelines read and write. Nothing was run.\n{lines}"
        )

    read_targets = [touch.target for plan in members.values() for touch in plan.reads]
    results = tuple(
        (touch.target, name)
        for name, touch in written
        if not any(touch.target.matches(target) for target in read_targets)
    )
    return tuple(handoffs), tuple((target, tuple(names)) for target, names in outside.items()), results


def _before_all(members: Mapping[str, MemberPlan], name: str) -> frozenset[str]:
    found: set[str] = set()
    waiting = list(members[name].after)
    while waiting:
        one = waiting.pop()
        if one not in found:
            found.add(one)
            waiting.extend(members[one].after)
    return frozenset(found)


def _digest(members: Mapping[str, MemberPlan]) -> str:
    """A fingerprint of what a real run of the chain does: its order, each pipeline's writes and values."""
    text = json.dumps(
        [[name, list(plan.after), plan.digest, {k: str(v) for k, v in plan.given.items()}] for name, plan in members.items()],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode("utf8")).hexdigest()
