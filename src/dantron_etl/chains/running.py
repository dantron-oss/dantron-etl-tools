"""Running a chain: each pipeline for real, in order, side by side where the order allows.

Each pipeline runs in a process of its own, so the memory it held is given back when it ends and a
pipeline the system stops cannot take the others with it. At most ``max_parallel`` run at a time.
A pipeline that fails stops every pipeline that waits for it; the others carry on. Each pipeline's
run is saved as it is when run alone, and the chain's run is saved beside them.

Why: docs/design/chains.md#running-a-chain
"""

from __future__ import annotations

import json
import multiprocessing
import os
import secrets
from collections.abc import Callable, Collection, Mapping
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..trace import StepEvent
from ..viewer.project import Project, ProjectError, combine_rules, load_project, own_rules, read_entry
from ..viewer.runs import RunRecord, _make_folder, _now, _owner_only, execute, writes, writes_digest
from .chain_file import ChainError, read_chain_file, value_for
from .plan import ChainPlan, load_chain, open_project

__all__ = [
    "CHAIN_RUNS_SUFFIX",
    "ChainRun",
    "MemberResult",
    "chain_runs",
    "run_chain",
    "run_member",
]

#: A chain's runs are saved in ``<runs folder>/<chain><suffix>/``. A pipeline name has no dot, so this
#: folder can never be a pipeline's.
CHAIN_RUNS_SUFFIX = ".chain"

#: What became of one pipeline in a chain's run.
FINISHED, FAILED, REFUSED, NOT_STARTED, NOT_IN_RUN = "finished", "failed", "refused", "not started", "not in this run"
_STOPPED_THE_CHAIN = (FAILED, REFUSED, NOT_STARTED)


@dataclass(frozen=True)
class MemberResult:
    """What one pipeline did in a chain's run."""

    pipeline: str
    outcome: str
    started: str | None = None
    finished: str | None = None
    steps_ran: int = 0
    steps_total: int = 0
    failed_step: str | None = None
    message: str | None = None
    #: The folder its run was saved in, while that run is still the one kept there.
    directory: str | None = None


@dataclass(frozen=True)
class ChainRun:
    """A chain's run: how it ended, the values it used, and what each pipeline did."""

    chain: str
    title: str
    outcome: str
    started: str
    finished: str
    values: Mapping[str, Any]
    covered: tuple[str, ...]
    members: tuple[MemberResult, ...]
    saved: Path

    def member(self, name: str) -> MemberResult:
        return next(result for result in self.members if result.pipeline == name)


def run_chain(
    where: Project | Path | str,
    name: str,
    *,
    confirm_digest: str,
    parameters: Mapping[str, Any] | None = None,
    start_from: str | Collection[str] | None = None,
    until: str | Collection[str] | None = None,
    report: Callable[[str], None] | None = None,
    executor: Callable[[int], Executor] | None = None,
    echo_steps: bool = True,
) -> ChainRun:
    """Run a chain for real and save the run. Returns how each pipeline did.

    ``confirm_digest`` must equal the digest of the plan that was shown (``load_chain(...).digest``);
    if any pipeline, the chain file or a value changed since, the run is refused with a
    :class:`ChainError` before anything runs. ``parameters``, ``start_from`` and ``until`` are as for
    :func:`load_chain` and :meth:`ChainPlan.select`. ``report`` is called with one line each time a
    pipeline starts or ends. ``executor`` makes the pool the pipelines run in, given how many may run
    at once; by default each runs in a new process. ``echo_steps`` prints each step's line as it ends.
    """
    project = load_project(open_project(where).root)
    plan = load_chain(project, name, parameters)
    if confirm_digest != plan.digest:
        raise ChainError(
            f"What chain '{name}' runs or writes has changed since it was shown, or it was never shown. "
            f"Look at the plan again and confirm that. Nothing was run."
        )
    covered = plan.select(start_from, until)
    say = report or (lambda line: None)
    make = executor or _process_pool
    started = _now()
    results: dict[str, MemberResult] = {}
    running: dict[Future[dict[str, Any]], str] = {}
    pool = make(plan.max_parallel)
    try:
        while True:
            for member in covered:
                if member in results or member in running.values():
                    continue
                failed_before = [one for one in plan.after(member) if results.get(one, _OK).outcome in _STOPPED_THE_CHAIN]
                if failed_before:
                    results[member] = MemberResult(
                        member, NOT_STARTED, message=f"not started, because {failed_before[0]} did not finish"
                    )
                    say(f"Not started: {member}, because {failed_before[0]} did not finish.")
            for member in covered:
                if len(running) >= plan.max_parallel:
                    break
                if member in results or member in running.values():
                    continue
                if all(one not in covered or results.get(one, _WAITING).outcome == FINISHED for one in plan.after(member)):
                    say(f"Starting {member}.")
                    job = (str(project.root), name, member, dict(plan.members[member].given), plan.members[member].digest, echo_steps)
                    try:
                        running[pool.submit(_run_one, *job)] = member
                    except BrokenProcessPool:
                        pool.shutdown(wait=False, cancel_futures=True)
                        pool = make(plan.max_parallel)
                        running[pool.submit(_run_one, *job)] = member
            if not running:
                break
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                member = running.pop(future)
                results[member] = _result(member, future)
                say(_ended(results[member]))
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)
        for future, member in running.items():
            results[member] = MemberResult(member, FAILED, message="stopped: the chain's run was interrupted")
        _save(plan, covered, results, started, "stopped")
        raise
    pool.shutdown(wait=True)
    outcome = FINISHED if all(results[member].outcome == FINISHED for member in covered) else FAILED
    return _save(plan, covered, results, started, outcome)


_OK = MemberResult("", FINISHED)
_WAITING = MemberResult("", "waiting")


def _process_pool(size: int) -> Executor:
    """A pool that runs each pipeline in a new process, started afresh rather than copied."""
    return ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"), max_tasks_per_child=1)


def _run_one(root: str, chain: str, member: str, given: dict[str, Any], digest: str, echo: bool) -> dict[str, Any]:
    """Run one pipeline of a chain for real. Runs in a worker: takes and returns plain values only."""
    project = load_project(Path(root))
    entry = project.pipelines[member]
    sink = _echo(member) if echo else None
    try:
        record = execute(project, entry, "real", sink=sink, confirm_digest=digest, given=given, chain=chain)
    except ProjectError as refusal:
        return {"outcome": REFUSED, "message": str(refusal)}
    return _record_fields(record)


def _echo(member: str) -> Callable[[StepEvent], None]:
    def show(event: StepEvent) -> None:
        rows = ", ".join(f"{shape.port.port}: {shape.rows} rows" for shape in event.outputs) or "no rows out"
        if len(event.outputs) == 1:
            rows = f"{event.outputs[0].rows} rows out"
        print(f"  [{member}] {event.step_id} ({event.tool}): {rows}, {event.duration_ms:.0f} ms", flush=True)

    return show


def _record_fields(record: RunRecord) -> dict[str, Any]:
    return {
        "outcome": record.outcome,
        "started": record.started,
        "finished": record.finished,
        "steps_ran": record.steps_ran,
        "steps_total": record.steps_total,
        "failed_step": record.failed_step,
        "message": record.message,
        "directory": str(record.directory),
    }


def _result(member: str, future: Future[dict[str, Any]]) -> MemberResult:
    try:
        fields = future.result()
    except BrokenProcessPool:
        return MemberResult(
            member,
            FAILED,
            message="the process running it stopped without a word. The usual cause is that the computer ran "
            "out of memory: lower max_parallel in the chain file.",
        )
    except Exception as error:  # noqa: BLE001 -- any failure of one pipeline is recorded, never raised
        return MemberResult(member, FAILED, message=f"{type(error).__name__}: {error}")
    return MemberResult(member, **fields)


def _ended(result: MemberResult) -> str:
    if result.outcome == FINISHED:
        return f"Finished: {result.pipeline}, {result.steps_ran} of {result.steps_total} steps."
    if result.outcome == FAILED and result.failed_step:
        return f"Failed: {result.pipeline} stopped at step {result.failed_step}: {result.message}"
    return f"{result.outcome.capitalize()}: {result.pipeline}: {result.message}"


# ---------------------------------------------------------------------------
# One pipeline of a chain, run by a scheduler
# ---------------------------------------------------------------------------


def run_member(
    where: Project | Path | str,
    chain: str,
    pipeline: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    logger: Any = None,
) -> RunRecord:
    """Run one pipeline of a chain for real with the chain's parameter values, and save the run.

    For a scheduler that runs each pipeline as a task of its own. Check the whole chain with
    :func:`load_chain` first: this runs the one pipeline and checks only that it is in the chain and
    that the values fit. ``logger`` receives each step's message (anything with ``info``, ``warning``
    and ``error``). Returns the run's record; its ``outcome`` is ``"finished"`` when it worked.
    """
    from ..runner import logging_sink

    project = open_project(where)
    if chain not in project.chains:
        raise ChainError(f"There is no chain {chain!r} in the project at {project.root}.")
    chain_file = read_chain_file(project.chains[chain])
    if pipeline not in {member.name for member in chain_file.members}:
        raise ChainError(f"{pipeline!r} is not a pipeline of chain '{chain}'.")
    asked = dict(parameters or {})
    for key in asked:
        if key not in chain_file.parameters:
            raise ChainError(f"A value was given for {key!r}, which is not a parameter of chain '{chain}'.")
    shown = f"chain '{chain}'"
    values = {
        key: value_for(key, default, asked[key], shown) if key in asked else default
        for key, default in chain_file.parameters.items()
    }
    entry = project.pipelines.get(pipeline)
    if entry is None:
        raise ChainError(f"Chain '{chain}' runs {pipeline!r}, which is not a pipeline in the project at {project.root}.")
    given = {key: value for key, value in values.items() if key in entry.rules_needed}
    spec = read_entry(project, entry, combine_rules(entry, own_rules(project, entry), given))
    digest = writes_digest(writes(spec, project.root))
    sink = logging_sink(logger) if logger is not None else None
    return execute(project, entry, "real", sink=sink, confirm_digest=digest, given=given, chain=chain)


# ---------------------------------------------------------------------------
# Saving a chain's run
# ---------------------------------------------------------------------------


def _save(
    plan: ChainPlan, covered: tuple[str, ...], results: Mapping[str, MemberResult], started: str, outcome: str
) -> ChainRun:
    stopped = MemberResult("", NOT_STARTED, message="not started: the chain's run stopped first")
    members = tuple(
        results.get(name) or (MemberResult(name, NOT_STARTED, message=stopped.message) if name in covered else MemberResult(name, NOT_IN_RUN))
        for name in plan.order
    )
    folder = plan.project.runs_folder / f"{plan.name}{CHAIN_RUNS_SUFFIX}"
    _make_folder(folder)
    stamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    target = folder / f"{stamp}-{secrets.token_hex(3)}.json"
    run = ChainRun(
        chain=plan.name,
        title=plan.title,
        outcome=outcome,
        started=started,
        finished=_now(),
        values=json.loads(json.dumps(dict(plan.values), default=str)),
        covered=covered,
        members=members,
        saved=target,
    )
    data = {key: value for key, value in asdict(run).items() if key != "saved"}
    incoming = folder / f".{target.name}.incoming"
    incoming.write_text(json.dumps(data, indent=2), encoding="utf8")
    _owner_only(incoming, 0o600)
    os.replace(incoming, target)
    return run


def chain_runs(project: Project, name: str) -> tuple[ChainRun, ...]:
    """A chain's saved runs, the latest first. A file that does not read back is skipped."""
    folder = project.runs_folder / f"{name}{CHAIN_RUNS_SUFFIX}"
    if not folder.is_dir():
        return ()
    found: list[ChainRun] = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf8"))
            members = tuple(MemberResult(**member) for member in data.pop("members"))
            covered = tuple(data.pop("covered"))
            found.append(ChainRun(**data, covered=covered, members=members, saved=path))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return tuple(found)
