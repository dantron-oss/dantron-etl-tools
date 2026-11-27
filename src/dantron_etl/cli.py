"""The command: one name, and a word after it saying what to do.

``view`` opens the viewer, ``run`` runs a pipeline or a chain, ``check`` loads one without running
it, ``chain suggest`` writes a first chain file, ``new`` writes a starter project. The rest print
the catalog.

This module imports only the standard library at the top, so ``--help`` answers quickly. Each
command imports what it needs when it runs.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .paths import ProjectRootNotFound

if TYPE_CHECKING:  # types only: each command imports what it needs when it runs
    from .chains import ChainPlan, ChainRun
    from .trace import StepEvent
    from .viewer.project import PipelineEntry, Project
    from .viewer.runs import RunRecord

__all__ = ["COMMAND", "browser_opener", "choose_port", "main"]

PACKAGE = __package__.split(".")[0]
COMMAND = PACKAGE.replace("_", "-")
VIEWER_APP = f"{COMMAND} viewer"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8899
SPARE_PORTS = range(8900, 8920)
LOOPBACK = ("127.0.0.1", "localhost", "::1", "[::1]")
ANY_ADDRESS = ("0.0.0.0", "::", "")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command with ``argv`` (the process arguments when None). Returns the exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["migrate"]:
        return _migrate(arguments[1:])
    parser = _parser()
    parsed = parser.parse_args(arguments)
    if parsed.action is None:
        parser.print_help()
        return 2
    return parsed.action(parsed)


# ---------------------------------------------------------------------------
# The parser and each command's help
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        description="Build, check, run and look at data pipelines written as YAML files.",
        epilog=f"Start here:  {COMMAND} new my-project   then   cd my-project   then   {COMMAND} view\n"
        f"Help on one command:  {COMMAND} <command> --help",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{COMMAND} {__version__}")
    parser.set_defaults(action=None)
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    def command(name: str, summary: str, description: str, example: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(
            name,
            help=summary,
            description=f"{summary}\n\n{description}",
            epilog=f"Example:\n  {example}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        return sub

    view = command(
        "view",
        "Open the viewer for the project you are in.",
        "Finds the project from the folder you are in, starts the viewer and opens it in your browser.\n"
        "The viewer lists every pipeline. From there you can run a preview and look at every step's rows.\n"
        "It keeps running until you press ctrl-c. If it is already running for this project, it is\n"
        "opened again instead of starting a second one.",
        f"{COMMAND} view",
    )
    view.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"the port to use (default {DEFAULT_PORT})")
    view.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"the address to listen on (default {DEFAULT_HOST}, this computer only)",
    )
    view.add_argument("--no-browser", action="store_true", help="do not open the browser; just print the address")
    view.set_defaults(action=_view)

    run = command(
        "run",
        "Run a pipeline, or a chain of pipelines.",
        "A pipeline runs as a preview by default: every step runs except the ones that write files or\n"
        "tables, and the steps after them. Nothing is written. The run is saved so the viewer can show it.\n"
        "With --write, it lists everything the run will write and asks you to type the pipeline's\n"
        "name before it goes ahead.\n\n"
        "A chain is checked and its order printed by default; nothing runs. With --write, it lists what\n"
        "every pipeline in it will write, asks you to type the chain's name, and runs them for real,\n"
        "side by side where the order allows. --set, --from and --until are for chains only.",
        f"{COMMAND} run month_end --write --set period=2026-12",
    )
    run.add_argument("pipeline", help="the pipeline's or chain's name, or the path to a pipeline's .yaml file")
    run.add_argument("--write", action="store_true", help="run every step, including the ones that write")
    run.add_argument("--yes", action="store_true", help="with --write, do not ask for the name")
    run.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE", help="a chain: change a parameter for this run"
    )
    run.add_argument(
        "--from", dest="start_from", action="append", metavar="PIPELINE",
        help="a chain: run this pipeline and every one after it",
    )
    run.add_argument(
        "--until", action="append", metavar="PIPELINE", help="a chain: run this pipeline and every one before it"
    )
    run.set_defaults(action=_run)

    chain = command(
        "chain",
        "Write a first chain file from what your pipelines read and write.",
        "'chain suggest NAME' writes chains/NAME.yaml. Each pipeline is set to run after the pipelines\n"
        "that write a file or table it reads. Name the pipelines to put in it, or leave them out for\n"
        "every pipeline in the project. Check the file before you run it.",
        f"{COMMAND} chain suggest month_end load_sources clean_ledger revenue --set period=2026-11",
    )
    chain_commands = chain.add_subparsers(title="what to do", metavar="suggest")
    suggest = chain_commands.add_parser("suggest", help="write a first chain file")
    suggest.add_argument("name", help="the chain's name: the file is chains/NAME.yaml")
    suggest.add_argument("pipelines", nargs="*", help="the pipelines to put in it (default: all of them)")
    suggest.add_argument(
        "--set", action="append", default=[], metavar="NAME=VALUE",
        help="a parameter of the chain, handed to every pipeline that uses ${rules.NAME}",
    )
    suggest.add_argument("--print", action="store_true", help="print the file instead of writing it")
    suggest.set_defaults(action=_suggest)
    chain.set_defaults(action=lambda arguments: _refuse(f"Say what to do: {COMMAND} chain suggest NAME"))

    check = command(
        "check",
        "Check a pipeline file without running it.",
        "Loads the file the way a run would and prints every problem found: a misspelt tool, a\n"
        "missing setting, a rule with no value, a step whose rows nothing uses. Prints 'ok' when\n"
        "there is none. Nothing is read from your data and nothing is written.",
        f"{COMMAND} check pipelines/example.yaml",
    )
    check.add_argument("file", help="the pipeline file to check")
    check.add_argument(
        "--rules",
        metavar="FILE",
        help="a YAML file of rule values, for a file that is not one of this project's pipelines",
    )
    check.set_defaults(action=_check)

    tools = command(
        "tools",
        "List every tool, grouped by what it does.",
        "One line per tool. Use the help command for everything about one tool.",
        f"{COMMAND} tools",
    )
    tools.set_defaults(action=_tools)

    help_ = command(
        "help",
        "Show everything about one tool.",
        "What the tool does, what it reads and gives, every setting, and a worked example\n"
        "with settings you can paste into a pipeline file.",
        f"{COMMAND} help join",
    )
    help_.add_argument("tool", help="the tool's name, as the tools command lists it")
    help_.set_defaults(action=_help)

    functions = command(
        "functions",
        "List the functions a [column] expression may use.",
        "Each function with what it gives and one example.",
        f"{COMMAND} functions",
    )
    functions.set_defaults(action=_functions)

    catalog = command(
        "catalog",
        "Write the tool catalog as a page you can open in a browser.",
        "Writes tool_catalog.html and tool_catalog.json. By default they go in the project's\n"
        "output_files folder, or in the folder you are in when it is not inside a project.",
        f"{COMMAND} catalog --out catalog",
    )
    catalog.add_argument("--out", metavar="DIR", help="the folder to write into")
    catalog.set_defaults(action=_catalog)

    command(
        "migrate",
        "Rewrite older pipeline files in the current tool and setting names.",
        "Without --write it only reports what would change. Comments and layout are kept.\n"
        "Every option after 'migrate' is passed on; see 'migrate --help'.",
        f"{COMMAND} migrate --write pipelines",
    )

    new = command(
        "new",
        "Make a new project folder with a working example.",
        "Writes a settings file, an example pipeline with its data, and launchers that open the\n"
        "viewer with a double-click. The folder must be new or empty.",
        f"{COMMAND} new my-project",
    )
    new.add_argument("folder", help="the folder to make")
    new.set_defaults(action=_new)
    return parser


def _refuse(message: object) -> int:
    if isinstance(message, ProjectRootNotFound):
        message = (
            f"This folder is not inside a project. Go to a project folder, or make one with: "
            f"{COMMAND} new my-project\n({message})"
        )
    print(message, file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# view
# ---------------------------------------------------------------------------


def _view(arguments: argparse.Namespace) -> int:
    from .viewer.project import ProjectError, find_project

    try:
        project = find_project(Path.cwd())
    except (ProjectRootNotFound, ProjectError) as error:
        return _refuse(error)
    os.chdir(project.root)
    host = arguments.host
    if host not in LOOPBACK:
        print(
            f"Warning: the viewer is listening on {host or 'every address'}. Anyone who can reach this "
            f"computer on the network can open it and read every row of every run."
        )
    # On this computer, or on every address: the viewer is reached, and opened, at 127.0.0.1.
    reached_locally = host in ANY_ADDRESS or host in LOOPBACK
    probe_host = DEFAULT_HOST if reached_locally else host
    port, running = choose_port(project.root, preferred=arguments.port, host=probe_host)
    if running:
        url = f"http://{DEFAULT_HOST}:{port}/"
        print(f"A viewer for {project.root} is already running at {url}")
        if not arguments.no_browser:
            _open_browser(url)
        return 0

    from .viewer.server import start_server

    try:
        server = start_server(project, host=host, port=port)
    except OSError as error:
        return _refuse(f"The viewer could not start on port {port}: {error}")
    url = f"http://{DEFAULT_HOST}:{server.port}/" if reached_locally else server.url
    if not arguments.no_browser:
        _open_browser(url)
    print(f"Viewer for {project.root} is running at {url}")
    print("Press ctrl-c to stop it.", flush=True)
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    print("The viewer has stopped.")
    return 0


def choose_port(root: Path, *, preferred: int = DEFAULT_PORT, host: str = DEFAULT_HOST) -> tuple[int, bool]:
    """The port to serve on, and whether a viewer for ``root`` already runs there.

    The preferred port when it is free. A port of 8900 to 8919 when a viewer for this project
    already holds it or the preferred one. Otherwise the first free port of those, then any free port.
    """
    if _can_bind(host, preferred):
        return preferred, False
    if _viewer_for(root, host, preferred):
        return preferred, True
    free = None
    for port in SPARE_PORTS:
        if port == preferred:
            continue
        if _can_bind(host, port):
            free = port if free is None else free
        elif _viewer_for(root, host, port):
            return port, True
    if free is not None:
        return free, False
    with socket.socket(_family(host), socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1], False


def _family(host: str) -> socket.AddressFamily:
    """IPv6 for an address with a colon in it, IPv4 for any other."""
    return socket.AF_INET6 if ":" in host else socket.AF_INET


def _can_bind(host: str, port: int) -> bool:
    """Whether a server could listen on ``host``:``port`` now."""
    with socket.socket(_family(host), socket.SOCK_STREAM) as probe:
        if os.name != "nt":
            # Lets a port a stopped viewer just left be used again. It never shares a port in use.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError:
            return False
        return True


def _viewer_for(root: Path, host: str, port: int) -> bool:
    """Whether the server on ``port`` is a viewer for the project at ``root``. Proxy settings are ignored."""
    import json

    shown = f"[{host}]" if ":" in host else host
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://{shown}:{port}/api/whoami", timeout=1) as response:
            answer = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(answer, Mapping) or answer.get("app") != VIEWER_APP:
        return False
    project = answer.get("project")
    return isinstance(project, str) and Path(project).resolve() == Path(root).resolve()


def browser_opener(environ: Mapping[str, str], platform_name: str, release: str) -> list[str] | None:
    """The program that opens an address in the browser on this machine, or None when there is none.

    Under WSL the Windows browser is used, display or not. On Linux a browser needs a display, so a
    container without one gets None.
    """
    if platform_name == "win32":
        return ["explorer.exe"]
    if platform_name == "darwin":
        return ["open"]
    if "microsoft" in release.lower() or environ.get("WSL_DISTRO_NAME"):
        return ["wslview"] if shutil.which("wslview") else ["explorer.exe"]
    if environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"):
        return ["xdg-open"]
    return None


def _open_browser(url: str) -> None:
    """Open ``url`` in the browser. Never fails: says what to do by hand when it cannot."""
    opener = browser_opener(os.environ, sys.platform, platform.release())
    if opener is not None:
        try:
            subprocess.Popen(
                [*opener, url],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        except OSError:
            pass
    print("Open the address below in your browser.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _run(arguments: argparse.Namespace) -> int:
    from .viewer.project import ProjectError, find_project, load_entry
    from .viewer.runs import execute, writes, writes_digest

    try:
        project = find_project(Path.cwd())
        if arguments.pipeline in project.chains:
            return _run_chain(arguments, project)
        entry = _entry(project, arguments.pipeline)
    except (ProjectRootNotFound, ProjectError) as error:
        return _refuse(error)
    chain_only = [option for option, given in (("--set", arguments.set), ("--from", arguments.start_from),
                  ("--until", arguments.until)) if given]
    if chain_only:
        verb = "is" if len(chain_only) == 1 else "are"
        return _refuse(f"{' and '.join(chain_only)} {verb} for a chain, and {entry.name} is a pipeline.")

    if not arguments.write:
        print(f"Preview of {entry.name}: the steps that write are held back.")
        record = execute(project, entry, "preview", sink=_print_event)
        return _report(record)

    try:
        spec = load_entry(project, entry)
    except Exception as error:  # noqa: BLE001 -- any refusal while loading is printed, not raised
        return _refuse(f"{entry.name} could not be loaded: {error}")
    found = writes(spec, project.root)
    if found:
        print(f"Running {entry.name} will write:")
        for write in found:
            print(f"  {write.step} ({write.tool}): {write.target}  [{write.detail}]")
    else:
        print(f"{entry.name} writes nothing.")
    if not arguments.yes:
        try:
            typed = input(f"Type the pipeline's name ({entry.name}) to go ahead: ").strip()
        except EOFError:
            typed = ""
        if typed != entry.name:
            return _refuse(f"Nothing was run: you typed {typed!r}, not {entry.name!r}.")
    try:
        record = execute(project, entry, "real", sink=_print_event, confirm_digest=writes_digest(found))
    except ProjectError as error:
        return _refuse(error)
    return _report(record)


def _entry(project: Project, asked: str) -> PipelineEntry:
    """The project's pipeline named ``asked``, or whose file ``asked`` points at."""
    from .errors import name_choices
    from .viewer.project import ProjectError

    if asked in project.pipelines:
        return project.pipelines[asked]
    path = Path(asked).resolve()
    for entry in project.pipelines.values():
        if entry.path == path:
            return entry
    chains = f" Its chains: {', '.join(project.chains)}." if project.chains else ""
    raise ProjectError(
        f"There is no pipeline {asked!r} in the project at {project.root}; "
        f"{name_choices(asked, project.pipelines, 'pipelines')}.{chains}"
    )


def _settings(given: Sequence[str]) -> dict[str, str]:
    """``NAME=VALUE`` texts as a mapping. The values stay text; the chain reads each as its kind."""
    from .chains import ChainError

    found: dict[str, str] = {}
    for text in given:
        name, equals, value = text.partition("=")
        if not equals or not name.strip():
            raise ChainError(f"--set {text!r} is not NAME=VALUE. Write it as, for example, --set period=2026-12")
        found[name.strip()] = value
    return found


def _run_chain(arguments: argparse.Namespace, project: Project) -> int:
    """Check a chain and print its plan; with --write, confirm and run it."""
    from .chains import ChainError, load_chain, run_chain

    name = arguments.pipeline
    try:
        plan = load_chain(project, name, _settings(arguments.set))
        covered = plan.select(arguments.start_from, arguments.until)
    except ChainError as error:
        return _refuse(error)
    print(_plan_text(plan, covered))
    if not arguments.write:
        print(f"Nothing was run. To run it for real: {COMMAND} run {name} --write")
        return 0
    print("Running it will write:")
    for member in covered:
        for write in plan.members[member].write_list:
            print(f"  {member}: {write.target}  [{write.detail}]")
    if not arguments.yes:
        try:
            typed = input(f"Type the chain's name ({name}) to go ahead: ").strip()
        except EOFError:
            typed = ""
        if typed != name:
            return _refuse(f"Nothing was run: you typed {typed!r}, not {name!r}.")
    try:
        run = run_chain(
            project, name, parameters=_settings(arguments.set), confirm_digest=plan.digest,
            start_from=arguments.start_from, until=arguments.until, report=lambda line: print(line, flush=True),
        )
    except ChainError as error:
        return _refuse(error)
    return _chain_report(run)


def _plan_text(plan: ChainPlan, covered: Sequence[str]) -> str:
    """A chain's plan in plain lines: values, order, hand-offs, and what comes from and goes outside."""
    lines = [f"Chain {plan.name}: {plan.title}"]
    if plan.values:
        lines.append("Parameters:")
        for key, value in plan.values.items():
            changed = " (given for this run)" if key in plan.asked else ""
            users = [name for name, member in plan.members.items() if key in member.given]
            lines.append(f"  {key}: {value}{changed}, used by {', '.join(users)}")
            for name in users:
                if key in plan.members[name].replaced:
                    lines.append(f"    replaces {name}'s own value {plan.members[name].replaced[key]}")
    at_once = "1 pipeline at a time" if plan.max_parallel == 1 else f"up to {plan.max_parallel} pipelines at a time"
    lines.append(f"Runs {at_once}, in this order:")
    width = max(len(name) for name in plan.order)
    for name in plan.order:
        after = f"after {', '.join(plan.after(name))}" if plan.after(name) else "first"
        skipped = "" if name in covered else "   (not in this run)"
        lines.append(f"  {name.ljust(width)}  {after}{skipped}")
    if plan.handoffs:
        lines.append("What passes between them:")
        lines += [f"  {one.writer} -> {one.reader}: {one.target.shown}" for one in plan.handoffs]
    if plan.outside:
        lines.append("Read from outside the chain:")
        lines += [f"  {target.shown} ({', '.join(readers)})" for target, readers in plan.outside]
    if plan.results:
        lines.append("Written and not read again inside the chain:")
        lines += [f"  {target.shown} ({writer})" for target, writer in plan.results]
    return "\n".join(lines)


def _chain_report(run: ChainRun) -> int:
    """Print how a chain's run ended and where it is saved. Returns the exit code."""
    counted: dict[str, list[str]] = {}
    for member in run.members:
        if member.pipeline in run.covered:
            counted.setdefault(member.outcome, []).append(member.pipeline)
    parts = [f"{len(names)} {outcome} ({', '.join(names)})" for outcome, names in counted.items()]
    stream = sys.stdout if run.outcome == "finished" else sys.stderr
    print(f"Chain {run.chain} {run.outcome}: {'; '.join(parts)}.", file=stream)
    print(f"The chain's run is saved in {run.saved}. To look at it: {COMMAND} view")
    return 0 if run.outcome == "finished" else 1


def _suggest(arguments: argparse.Namespace) -> int:
    """Write, or print, a first chain file worked out from what the pipelines read and write."""
    from .chains import ChainError, suggest_chain
    from .chains.chain_file import read_value
    from .viewer.project import PIPELINE_NAME, ProjectError, find_project

    try:
        project = find_project(Path.cwd())
        if not PIPELINE_NAME.fullmatch(arguments.name):
            raise ChainError(f"{arguments.name!r} is not a chain name. Use letters, digits, '_' and '-'.")
        if arguments.name in project.pipelines:
            raise ChainError(f"A pipeline is already called {arguments.name}. Give the chain another name.")
        # With no default to match, a value is read the way the chain file will read it back.
        values = {key: read_value(text) for key, text in _settings(arguments.set).items()}
        text = suggest_chain(project, arguments.name, arguments.pipelines, values)
    except (ProjectRootNotFound, ProjectError) as error:
        return _refuse(error)
    if arguments.print:
        print(text, end="")
        return 0
    folder = project.chain_folders[0]
    target = folder / f"{arguments.name}.yaml"
    if target.exists():
        return _refuse(f"{target} is already there. Choose another name, or use --print to see the suggestion.")
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf8")
    print(f"Wrote {target}. Read it, then: {COMMAND} run {arguments.name}")
    return 0


def _print_event(event: StepEvent) -> None:
    """One line for a step that ended: its rows out and how long it took."""
    shapes = event.outputs
    if len(shapes) == 1:
        rows = f"{shapes[0].rows} rows out"
    else:
        rows = ", ".join(f"{shape.port.port}: {shape.rows} rows" for shape in shapes) or "no rows out"
    print(f"  {event.step_id} ({event.tool}): {rows}, {event.duration_ms:.0f} ms", flush=True)


def _report(record: RunRecord) -> int:
    """Print how a run ended and where it is saved. Returns the exit code."""
    if record.held_back:
        print(f"Held back, nothing written: {', '.join(record.held_back)}")
    if record.outcome == "failed":
        print(f"The run stopped at step {record.failed_step}: {record.message}", file=sys.stderr)
    elif record.outcome == "refused":
        print(f"The pipeline did not run: {record.message}", file=sys.stderr)
    else:
        print(f"Finished: {record.steps_ran} of {record.steps_total} steps ran.")
    print(f"The run is saved in {record.directory}. To look at it: {COMMAND} view")
    return 0 if record.outcome == "finished" else 1


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _check(arguments: argparse.Namespace) -> int:
    from .expr import ExpressionError
    from .errors import ToolError
    from .loader import dead_ends, read_pipeline
    from .viewer.project import ProjectError, find_project, pipeline_rules
    from .viewer.project_settings import read_yaml, rules_from_file

    path = Path(arguments.file).resolve()
    if not path.is_file():
        return _refuse(f"There is no file {arguments.file}.")
    chain = _chain_at(path)
    if chain is not None:
        return _check_chain(*chain, shown=arguments.file)
    problems: list[str] = []
    try:
        document = read_yaml(path)
    except ProjectError as error:
        return _refuse(error)

    root, rules, rules_ok = Path.cwd(), None, True
    try:
        if arguments.rules:
            rules_path = Path(arguments.rules).resolve()
            if not rules_path.is_file():
                return _refuse(f"There is no rules file {arguments.rules}.")
            rules = rules_from_file(root, path.stem, str(rules_path))
        else:
            with contextlib.suppress(ProjectRootNotFound):
                project = find_project(path.parent)
                root = project.root
                entry = next((entry for entry in project.pipelines.values() if entry.path == path), None)
                if entry is not None:
                    rules = pipeline_rules(project, entry)
    except ProjectError as error:
        problems.append(str(error))
        rules_ok = False

    steps = 0
    if rules_ok:
        try:
            with contextlib.chdir(root):
                steps = len(read_pipeline(path, rules=rules).steps)
        except (ToolError, ExpressionError) as error:
            problems.append(str(error))
        except Exception as error:  # noqa: BLE001 -- a YAML error, for one, is a problem to print
            problems.append(f"{type(error).__name__}: {error}")
    if isinstance(document, Mapping):
        with contextlib.suppress(Exception):
            dead = dead_ends(document)
            if dead:
                problems.append(f"Steps whose rows nothing uses (dead ends): {', '.join(dead)}")

    if not problems:
        print(f"ok, {steps} steps")
        return 0
    print(f"{arguments.file}: {len(problems)} problem{'s' if len(problems) != 1 else ''}", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    return 1


def _chain_at(path: Path) -> tuple[Project, str] | None:
    """The project and chain name when ``path`` is one of a project's chain files."""
    from .viewer.project import ProjectError, find_project

    try:
        project = find_project(path.parent)
    except (ProjectRootNotFound, ProjectError):
        return None
    name = next((name for name, entry in project.chains.items() if entry.path == path), None)
    return None if name is None else (project, name)


def _check_chain(project: Project, name: str, *, shown: str) -> int:
    """Check a chain the way a run would, without running anything."""
    from .chains import ChainError, load_chain

    try:
        plan = load_chain(project, name)
    except ChainError as error:
        print(f"{shown}: the chain is refused", file=sys.stderr)
        print(error, file=sys.stderr)
        return 1
    print(f"ok, a chain of {len(plan.order)} pipelines")
    return 0


# ---------------------------------------------------------------------------
# The catalog, migrate and new
# ---------------------------------------------------------------------------


def _tools(arguments: argparse.Namespace) -> int:
    from .catalog import tools_text

    print(tools_text())
    return 0


def _help(arguments: argparse.Namespace) -> int:
    from .catalog import describe
    from .errors import ToolError

    try:
        print(describe(arguments.tool))
    except ToolError as error:
        return _refuse(error)
    return 0


def _functions(arguments: argparse.Namespace) -> int:
    from .catalog import functions_text

    print(functions_text())
    return 0


def _catalog(arguments: argparse.Namespace) -> int:
    from .viewer.catalog_page import write_catalog
    from .viewer.project import ProjectError, find_project

    if arguments.out:
        out = Path(arguments.out)
    else:
        try:
            out = find_project(Path.cwd()).root / "output_files"
        except (ProjectRootNotFound, ProjectError):
            out = Path.cwd()
    for path in write_catalog(out):
        print(f"Wrote {path}")
    return 0


def _migrate(rest: list[str]) -> int:
    from .migrate import main as migrate_main

    return migrate_main(rest)


def _new(arguments: argparse.Namespace) -> int:
    from .viewer.project import ProjectError
    from .viewer.scaffold import write_project

    folder = Path(arguments.folder)
    try:
        written = write_project(folder)
    except ProjectError as error:
        return _refuse(error)
    print(f"Made a new project in {folder.resolve()}:")
    for path in written:
        print(f"  {path.relative_to(folder).as_posix()}")
    print("Next:")
    print(f"  cd {folder}")
    print(f"  {COMMAND} view")
    return 0
