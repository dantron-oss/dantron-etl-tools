"""Reading a chain file: the pipelines it runs, what each runs after, and its parameters.

Everything here reads the file alone. Whether the pipelines exist, and whether the order agrees with
the files and tables they pass on, is checked in ``plan.py``.

Why: docs/design/chains.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ..viewer.project import ChainEntry, ProjectError
from ..viewer.project_settings import read_yaml
from ..viewer.project_settings import refuse_unknown as _refuse_unknown

__all__ = [
    "CHAIN_KEYS",
    "DEFAULT_MAX_PARALLEL",
    "MEMBER_KEYS",
    "ChainError",
    "ChainFile",
    "Member",
    "parse_chain",
    "read_chain_file",
    "read_value",
    "run_order",
    "value_for",
]

#: The keys a chain file may have at its top.
CHAIN_KEYS = ("title", "parameters", "max_parallel", "pipelines")
#: The keys each entry under ``pipelines`` may have.
MEMBER_KEYS = ("name", "after", "why")
#: How many pipelines run at the same moment when the chain file does not say.
DEFAULT_MAX_PARALLEL = 2

_EXAMPLE = "pipelines:\n  - name: load_sources\n  - name: clean_ledger\n    after: [load_sources]"


class ChainError(ProjectError):
    """A chain that cannot run as written. The message says what to change."""


@dataclass(frozen=True)
class Member:
    """One pipeline in a chain, and the pipelines it runs after."""

    name: str
    after: tuple[str, ...]
    why: str | None = None


@dataclass(frozen=True)
class ChainFile:
    """A chain file as written, checked on its own."""

    name: str
    path: Path
    title: str
    parameters: Mapping[str, Any]
    max_parallel: int
    members: tuple[Member, ...]

    def member(self, name: str) -> Member:
        return next(member for member in self.members if member.name == name)


def refuse_unknown(found: Mapping[Any, Any], allowed: tuple[str, ...], where: str) -> None:
    """Refuse a key that is not one of ``allowed``, as a ChainError."""
    try:
        _refuse_unknown(found, allowed, where)
    except ProjectError as error:
        raise ChainError(str(error)) from None


def read_chain_file(entry: ChainEntry) -> ChainFile:
    """Read and check the chain file a project found."""
    if entry.problem:
        raise ChainError(entry.problem)
    try:
        document = read_yaml(entry.path)
    except ProjectError as error:
        raise ChainError(str(error)) from None
    return parse_chain(document, entry.name, entry.path)


def parse_chain(document: Any, name: str, path: Path) -> ChainFile:
    """Check a chain document on its own and return it. Refuses the first problem, by name."""
    where = f"the chain file {path.name}"
    if not isinstance(document, Mapping):
        raise ChainError(f"{where} holds {type(document).__name__}, not a chain. Write it as:\n{_EXAMPLE}")
    refuse_unknown(document, CHAIN_KEYS, where)
    title = document.get("title", name)
    if not isinstance(title, str) or not title.strip():
        raise ChainError(f"'title' in {where} is {title!r}. Write a short name for the chain.")
    members = _members(document.get("pipelines"), where)
    _check_after(members, where)
    run_order(members, where)  # refuses a loop
    return ChainFile(
        name=name,
        path=path,
        title=title,
        parameters=MappingProxyType(_parameters(document.get("parameters"), where)),
        max_parallel=_max_parallel(document.get("max_parallel", DEFAULT_MAX_PARALLEL), where),
        members=members,
    )


def _members(value: Any, where: str) -> tuple[Member, ...]:
    if not isinstance(value, list) or not value:
        raise ChainError(
            f"'pipelines' in {where} is {value!r}. List the pipelines the chain runs, for example:\n{_EXAMPLE}"
        )
    members: list[Member] = []
    for at, item in enumerate(value, start=1):
        if isinstance(item, str):
            item = {"name": item}
        if not isinstance(item, Mapping):
            raise ChainError(
                f"Entry {at} under 'pipelines' in {where} is {item!r}. Write '- name: <pipeline>', "
                f"with 'after: [...]' when it runs after others."
            )
        refuse_unknown(item, MEMBER_KEYS, f"entry {at} under 'pipelines' in {where}")
        member_name = item.get("name")
        if not isinstance(member_name, str) or not member_name:
            raise ChainError(f"Entry {at} under 'pipelines' in {where} has no 'name'. Name the pipeline it runs.")
        after = item.get("after", [])
        if isinstance(after, str):
            after = [after]
        if not isinstance(after, list) or not all(isinstance(one, str) and one for one in after):
            raise ChainError(
                f"'after' for {member_name} in {where} is {after!r}. List pipeline names, for example "
                f"'after: [load_sources]'."
            )
        why = item.get("why")
        if why is not None and not isinstance(why, str):
            raise ChainError(f"'why' for {member_name} in {where} is {why!r}. Write a sentence.")
        members.append(Member(member_name, tuple(dict.fromkeys(after)), why))
    return tuple(members)


def _check_after(members: Sequence[Member], where: str) -> None:
    seen: set[str] = set()
    for member in members:
        if member.name in seen:
            raise ChainError(f"{member.name} is listed twice in {where}. A chain runs each pipeline once.")
        seen.add(member.name)
    for member in members:
        for before in member.after:
            if before == member.name:
                raise ChainError(f"{member.name} is set to run after itself in {where}. Take it out of its 'after'.")
            if before not in seen:
                raise ChainError(
                    f"{member.name} runs after {before} in {where}, but {before} is not in this chain. "
                    f"Add '- name: {before}' to the chain, or take it out of {member.name}'s 'after'."
                )


def run_order(members: Sequence[Member], where: str) -> tuple[str, ...]:
    """Every pipeline after all the ones it runs after. Ties keep the order of the file.

    Refuses a loop, naming the pipelines caught in it.
    """
    left = {member.name: set(member.after) for member in members}
    order: list[str] = []
    while left:
        ready = [member.name for member in members if member.name in left and not left[member.name]]
        if not ready:
            stuck = ", ".join(member.name for member in members if member.name in left)
            raise ChainError(
                f"The pipelines in {where} wait for each other in a loop, so none of these can start: "
                f"{stuck}. Take one of them out of another's 'after'."
            )
        name = ready[0]
        order.append(name)
        del left[name]
        for waiting in left.values():
            waiting.discard(name)
    return tuple(order)


def _parameters(value: Any, where: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ChainError(
            f"'parameters' in {where} is {value!r}. Write one 'name: default value' line per parameter, "
            f"for example\nparameters:\n  period: \"2026-11\""
        )
    found: dict[str, Any] = {}
    for key, default in value.items():
        if not isinstance(key, str) or not key or "}" in key:
            raise ChainError(f"The parameter {key!r} in {where} is not a name a pipeline can use as ${{rules.name}}.")
        if _kind(default) is None:
            raise ChainError(
                f"The parameter {key} in {where} has the default {default!r}. Give it a value: text, a whole "
                f"number, a number, true or false, a date, or a list of those."
            )
        found[key] = default
    return found


def _max_parallel(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ChainError(
            f"'max_parallel' in {where} is {value!r}. Write how many pipelines may run at the same moment: "
            f"a whole number, 1 or more. Leave it out for {DEFAULT_MAX_PARALLEL}."
        )
    return value


# ---------------------------------------------------------------------------
# Parameter values
# ---------------------------------------------------------------------------

_KIND_WORDS = {
    "text": "text",
    "whole number": "a whole number",
    "number": "a number",
    "true or false": "true or false",
    "date": "a date, written 2026-11-30",
    "date and time": "a date and time, written 2026-11-30 09:00:00",
    "list": "a list, written [A, B]",
}


def _kind(value: Any) -> str | None:
    """The kind of a parameter value, in plain words, or None for a value a parameter cannot hold."""
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, int):
        return "whole number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, datetime):
        return "date and time"
    if isinstance(value, date):
        return "date"
    if isinstance(value, list) and all(_kind(item) not in (None, "list") for item in value):
        return "list"
    return None


def value_for(name: str, default: Any, given: Any, where: str) -> Any:
    """The value ``given`` for a parameter, as the same kind as its default, or a refusal.

    Text from a command line or a scheduler's form is read the way YAML reads it, so ``2026-11-30``
    becomes a date where the default is a date. Where the default is text, the text is kept as typed.
    """
    wanted = _kind(default)
    value = given
    if isinstance(given, str) and wanted not in (None, "text"):
        value = read_value(given)
    if wanted == "number" and _kind(value) == "whole number":
        value = float(value)
    if _kind(value) != wanted:
        raise ChainError(
            f"The parameter {name} in {where} is {_KIND_WORDS[wanted or 'text']} (its default is {default!r}), "
            f"and the value given, {given!r}, is not. Give {_KIND_WORDS[wanted or 'text']}."
        )
    return value


def read_value(text: str) -> Any:
    """Text read as one YAML value. Text YAML cannot read stays text."""
    import yaml

    from ..loader import yaml_loader

    try:
        return yaml.load(text, Loader=yaml_loader())
    except (yaml.YAMLError, ValueError):
        return text
