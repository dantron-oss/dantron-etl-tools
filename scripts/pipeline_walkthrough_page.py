"""Run one pipeline, save the run, and show it as a page: the canvas, and the data at every port.

    .venv/bin/python scripts/pipeline_walkthrough_page.py <name>     # or: make run p=<name>

<name> picks a project: the flow in pipelines/<name>.yaml, its input files in pipelines/<name>/, and
its settings in pipelines/<name>/settings.yaml. The run is saved to output_files/runs/<name>/, the
viewer is started if it is not already running, and the page opens in a browser.

The page exists because a figure that is wrong has to be traceable to the step that made it
wrong. So it draws the flow as it is wired, and then shows, for every port, the rows that
actually left it -- including the ports a step set aside, which is where a missing figure
usually went.

Everything here is read from the pipeline file and the trace of a run. Nothing is written down
twice: the shape of the canvas comes from the edges, the panel under a step comes from that
step's own configuration as authored, and the tables come from the frames the run produced.
"""


from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dantron_etl
from dantron_etl import PipelineTrace, StepFailedError, ToolError, read_pipeline, run_pipeline
from dantron_etl.expr import ExpressionError
from dantron_etl.focus import FocusSpec
from dantron_etl.paths import project_root
from dantron_etl.preview import PAGE_NAME, PreviewError, persist_panels, persist_trace, serve
from dantron_etl.viewer import build, build_panels

# Resolved from this script's own location rather than from the library's -- see `dantron_etl.paths`.
PROJECT_ROOT = project_root(__file__)
PIPELINES_DIR = PROJECT_ROOT / "pipelines"
# Where a run is saved, one folder per project. The same folder scripts/serve_runs.py serves.
RUNS_ROOT = PROJECT_ROOT / "output_files" / "runs"
SETTINGS_NAME = "settings.yaml"
# The port the viewer listens on. The same one scripts/serve_runs.py uses, so either starts the
# viewer and both find it running.
VIEWER_PORT = 8899


def _refuse_a_borrowed_checkout() -> None:
    """Stop before rendering a page out of some other copy of this repository.

    This repository is worked on in git worktrees against one editable install that points at the
    main checkout. Run a script from a worktree without ``PYTHONPATH=<worktree>/src`` and
    ``dantron_etl`` resolves there instead: the page is then built by the main checkout's tools, and
    nothing in it says so. That has happened here once already -- a run that read a pipeline file the
    author had never touched and looked entirely ordinary on the way past. The project root is now
    resolved from this file rather than from an imported package, so it can only disagree with this
    checkout when ``ETL_PROJECT_ROOT`` points elsewhere -- which is the same mistake, spelled out.

    A page is a statement about one flow, so it has to be a statement about *this* one. The check is
    cheap, it names the fix, and it is worth more than trusting every future caller to remember.
    """
    here = Path(__file__).resolve().parents[1]
    # dantron_etl/__init__.py -> dantron_etl -> src -> the checkout it was installed from.
    resolved = {"the project root": PROJECT_ROOT, "dantron_etl": Path(dantron_etl.__file__).parents[2]}
    borrowed = {name: root for name, root in resolved.items() if root.resolve() != here}
    if borrowed:
        named = ", and ".join(f"{name} is {root}" for name, root in borrowed.items())
        raise SystemExit(
            f"refusing to run: this script is in {here}, {named}.\n"
            f"The page would be built from another checkout's pipelines and tools. Run it as:\n"
            f"  PYTHONPATH={here / 'src'} .venv/bin/python {Path(__file__).resolve().relative_to(here)}"
        )


class ProjectError(Exception):
    """A project that cannot be run as asked. The message says what to change, in plain words."""


@dataclass(frozen=True)
class Project:
    """What a name on the command line stands for: a flow, the values it needs, and who to follow."""

    name: str
    pipeline: Path
    rules: dict[str, Any]
    focus: FocusSpec


# A project name becomes a file name and a folder name, so it is kept to characters that are safe as
# both on every system. A slash or a ".." would point outside pipelines/ altogether.
_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

_SETTINGS_EXAMPLE = """  rules:                # the values the flow reads as ${rules.<name>}
    revenue_type: REVENUE
  focus: party          # the column that identifies one entity, to follow it through the flow"""


def projects(pipelines_dir: Path = PIPELINES_DIR) -> list[str]:
    """Every project name there is a pipeline file for."""
    return sorted(path.stem for path in pipelines_dir.glob("*.yaml"))


def load_project(name: str, pipelines_dir: Path = PIPELINES_DIR) -> Project:
    """The project ``name`` stands for, or a :class:`ProjectError` saying what is missing or wrong."""
    if not _PROJECT_NAME.fullmatch(name):
        raise ProjectError(
            f"'{name}' is not a project name. A project name is letters, digits, '_' and '-', and names "
            f"a file in {pipelines_dir}: 'sales' runs {pipelines_dir / 'sales.yaml'}."
        )
    pipeline = pipelines_dir / f"{name}.yaml"
    if not pipeline.is_file():
        known = ", ".join(projects(pipelines_dir)) or "none yet"
        raise ProjectError(f"there is no pipeline '{name}': '{pipeline}' does not exist. The projects there are: {known}.")

    settings_path = pipelines_dir / name / SETTINGS_NAME
    if not settings_path.is_file():
        raise ProjectError(
            f"project '{name}' has no settings file. Create '{settings_path}' with the values the flow "
            f"needs and the column that identifies one entity, like this:\n{_SETTINGS_EXAMPLE}"
        )
    settings = _read_settings(settings_path)
    return Project(name=name, pipeline=pipeline, rules=settings["rules"], focus=settings["focus"])


def _read_settings(path: Path) -> dict[str, Any]:
    """``rules`` and ``focus`` from a settings file, each checked, with any other key refused by name."""
    import yaml

    try:
        document = yaml.safe_load(path.read_text(encoding="utf8"))
    except yaml.YAMLError as error:
        raise ProjectError(f"'{path}' is not readable YAML: {error}") from error
    if document is None:
        document = {}
    if not isinstance(document, dict):
        raise ProjectError(f"'{path}' must hold 'rules' and 'focus', like this:\n{_SETTINGS_EXAMPLE}")

    # A misspelt key would otherwise be ignored, and the flow would then run without the value it
    # was meant to carry. Refused here, by name, instead.
    unknown = sorted(str(key) for key in document if key not in ("rules", "focus"))
    if unknown:
        raise ProjectError(f"'{path}' has {', '.join(repr(key) for key in unknown)}; a settings file holds only 'rules' and 'focus'.")

    rules = document.get("rules") or {}
    if not isinstance(rules, dict) or not all(isinstance(key, str) for key in rules):
        raise ProjectError(f"'rules' in '{path}' must be a list of names and values, like this:\n{_SETTINGS_EXAMPLE}")

    focus = document.get("focus")
    columns = [focus] if isinstance(focus, str) else focus
    if not columns or not isinstance(columns, list) or not all(isinstance(column, str) and column for column in columns):
        raise ProjectError(
            f"'{path}' must say which column identifies one entity, so the page can follow it "
            f"through the flow: 'focus: party', or a list such as 'focus: [account, customer]' to "
            f"take the first of them each step has."
        )
    return {"rules": dict(rules), "focus": FocusSpec(label=columns[0], identity=tuple(columns))}


def write_run(project: Project, runs_root: Path = RUNS_ROOT) -> tuple[Path, PipelineTrace]:
    """Run ``project`` and save it under ``runs_root``: its rows, its panels and its page.

    A step that fails stops the run. What ran before it is still saved, with the page, so the rows
    that went into the failing step can be looked at. The failure is then raised as it came.
    """
    spec = read_pipeline(project.pipeline, rules=project.rules)
    failure: StepFailedError | None = None
    try:
        trace = run_pipeline(spec)
    except StepFailedError as error:
        failure, trace = error, error.trace

    directory = runs_root / project.name
    # The previous run is removed first. Saving over it would leave the files of any step that has
    # since been renamed or removed, and the page would offer data this run never produced.
    if directory.exists():
        shutil.rmtree(directory)
    persist_trace(trace, directory)
    persist_panels(build_panels(spec, trace, project.focus), directory)
    (directory / PAGE_NAME).write_text(build(spec, trace, project.focus), encoding="utf-8")
    if failure is not None:
        raise failure
    return directory, trace


def _viewer_serves(name: str, port: int) -> bool:
    """Whether a viewer on ``port`` is already answering for the run called ``name``."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{name}/api/ports", timeout=2) as answer:
            return answer.status == 200 and "ports" in json.loads(answer.read())
    except (OSError, ValueError):
        return False


def open_in_browser(url: str) -> None:
    """Open ``url`` in the person's browser, including from WSL, where the browser is on Windows."""
    explorer = shutil.which("explorer.exe")
    if explorer is not None:
        # explorer.exe hands an address to the Windows default browser. It reports failure even when
        # it worked, so its exit code says nothing and is not checked.
        subprocess.run([explorer, url], check=False)
    elif not webbrowser.open(url):
        print(f"open {url} in a browser")


def view(name: str, runs_root: Path = RUNS_ROOT, port: int = VIEWER_PORT) -> None:
    """Show the saved run ``name``: in the viewer already running, or in one started here."""
    url = f"http://localhost:{port}/{name}/"
    if _viewer_serves(name, port):
        print(f"the viewer is already running; opening {url}")
        open_in_browser(url)
        return
    try:
        server = serve(runs_root, port=port, source_root=PROJECT_ROOT)
    except OSError as error:
        raise ProjectError(
            f"the viewer could not start on port {port}: {error}. Something else is using that port; "
            f"stop it, or look at the saved run with scripts/serve_runs.py once the port is free."
        ) from error
    print(f"viewing {url}\n\n  ctrl-c to stop")
    open_in_browser(url)
    # A sleeping loop rather than Event().wait(), so ctrl-c is heard. See scripts/serve_runs.py.
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    print("\nstopping")
    server.shutdown()
    server.server_close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a pipeline, save the run, and open its page.")
    parser.add_argument("project", help="the project to run: pipelines/<project>.yaml")
    parser.add_argument("--no-view", action="store_true", help="save the run without starting the viewer")
    args = parser.parse_args(argv)

    _refuse_a_borrowed_checkout()
    try:
        project = load_project(args.project)
        directory, trace = write_run(project, RUNS_ROOT)
    except StepFailedError as error:
        print(f"step '{error.step_id}' failed: {error}", file=sys.stderr)
        print(f"what ran before it is saved in '{RUNS_ROOT / args.project}'", file=sys.stderr)
        if not args.no_view:
            view(args.project, RUNS_ROOT)
        return 1
    except (ProjectError, ToolError, ExpressionError, PreviewError) as error:
        print(f"cannot run '{args.project}': {error}", file=sys.stderr)
        return 1

    print(f"{len(trace.events)} steps ran, {len(trace.shapes)} ports saved in '{directory}'")
    if not args.no_view:
        view(project.name, RUNS_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
