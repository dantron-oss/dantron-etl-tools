"""Tests for the tool library's own project-root convention, and for the viewer scripts using it.

The library is about to be lifted into its own distribution and imported by other clients, and the
last thread tying it to this repository is four viewer scripts that say ``from acme.paths import
PROJECT_ROOT``. Cutting that thread is not a rename, because the client module resolves its root by
walking up from its own file::

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

which is right while the package sits in the ``src/`` of the repository it belongs to, and silently
wrong the moment the same expression ships inside an installed library::

    /client/.venv/lib/python3.13/site-packages/<pkg>/paths.py  ->  parents[2]  ->  /client/.venv/lib/python3.13

Nothing raises. Every path afterwards resolves against a directory inside the virtualenv, which
exists, is readable, and is not the project. This repository has paid for that failure twice in one
day -- a validation workbook built from another checkout's pipeline file, and a run that read a
sixteen-step pipeline where a six-hundred-and-seventy-two-step one was meant -- and both looked
entirely ordinary on the way past. So the rulings tested here are about refusing, not resolving.

**The convention pinned below: environment variable, else search upward from a start the caller
names, else refuse.**

    from dantron_etl.paths import project_root
    ROOT = project_root(__file__)

Why not the two alternatives:

*Infer from the library's own ``__file__``.* That is the trap above, in the library where it does
the most damage: one wrong answer, reused by every consumer, never raised. Excluded by construction
here -- ``project_root`` takes the caller's location because the library has no honest opinion
about where the caller's project is, and ``test_an_installed_copy_refuses...`` re-enacts the
site-packages geometry to prove an installed copy cannot answer.

*Environment variable, else the current working directory.* Correct for a command-line tool invoked
from inside a project, and wrong for a library: the cwd of a long-running process, a Dagster daemon
or a notebook kernel is an accident of how it was launched, and it changes underneath a running
process. It also fails in the direction that hurts -- it answers plausibly instead of refusing.

*Environment variable, else search upward from the caller's file for a project marker.* Correct in a
repository, correct in a worktree, correct when the caller is a script and when it is an installed
module, and -- the reason it is chosen -- **refusable**: when no marker is found there is nothing
plausible to return, so the failure is a named exception rather than a path. The cost is that every
caller must pass ``__file__``. That is the cost being bought deliberately: an explicit argument at
four call sites against a class of silent defect that has already cost this project two rebuilds.

The environment variable stays, under the library's own name, because this repository is worked on
in git worktrees against one editable install and every real run sets a root override; a library
that could not be pointed at a root would be unusable here on day one.
"""

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from dantron_etl.paths import ROOT_ENV_VAR, ROOT_MARKERS, ProjectRootNotFound, project_root

# Anchored on this test file rather than on ``acme.paths.PROJECT_ROOT``, deliberately. The checkout
# under test is the one this file sits in; importing the client's answer to that question would make
# the tests agree with whichever checkout happened to be installed -- which is the defect, not the
# measurement of it.
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
LIBRARY_SRC = REPO_SRC / "dantron_etl"
SCRIPTS_DIR = REPO_ROOT / "scripts"

#: The literal every script, container and CI job will spell. Written out here rather than taken
#: from ``ROOT_ENV_VAR`` so that renaming the constant cannot quietly rename the contract.
ROOT_ENV_VAR_NAME = "ETL_PROJECT_ROOT"

#: The four viewer scripts that ship with the library and must stop importing client code.
LIBRARY_SCRIPTS = (
    "serve_runs.py",
    "pipeline_walkthrough_page.py",
    "tool_catalog_page.py",
    "etl_architecture_page.py",
)

CLIENT_PACKAGE = "acme"


# ---------------------------------------------------------------------------
# Subprocess probes
#
# Every claim about what a module imports, or about what it resolves at import time, is measured in
# a fresh interpreter. The pytest session has already imported acme (conftest.py does) and already
# has this repository's root on sys.path, so an in-process check of either would pass vacuously --
# the same reasoning the existing boundary test states for its own subprocess.
# ---------------------------------------------------------------------------

#: Import the library's paths module from a nominated sys.path entry and report every module-level
#: Path it holds that is an ancestor of its own file -- i.e. every root it inferred from __file__.
_LIBRARY_IMPORT_PROBE = """
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from dantron_etl import paths as library_paths

here = Path(library_paths.__file__).resolve()
inferred = {
    name: str(value)
    for name, value in vars(library_paths).items()
    if isinstance(value, Path) and here.is_relative_to(Path(value).resolve())
}
print(json.dumps({"module_file": str(here), "inferred": inferred}))
"""

#: Ask a library copy for a root, starting from its own installed location.
_INSTALLED_COPY_PROBE = """
import json, sys

sys.path.insert(0, sys.argv[1])
from dantron_etl import paths as library_paths

answer = {"module_file": library_paths.__file__}
try:
    answer["answered"] = str(library_paths.project_root(library_paths.__file__))
except Exception as exc:
    answer["refused"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(answer))
"""

#: Execute a script by path and report what it imported and every module-level Path it defined.
_SCRIPT_PROBE = """
import importlib.util, json, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("script_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

print(json.dumps({
    "top_level": sorted({name.split(".")[0] for name in sys.modules}),
    "paths": {name: str(value) for name, value in vars(module).items() if isinstance(value, Path)},
}))
"""


def _probe(body: str, argument: str, *, cwd: Path, env_overrides: Mapping[str, str | None] = {}) -> dict:
    """Run a probe in a fresh interpreter and return the JSON object it printed.

    ``PYTHONPATH`` points at this checkout's ``src`` so that the subprocess reads the code being
    tested rather than whichever checkout the editable install happens to point at -- the same
    borrowed-checkout hazard the module under test exists to refuse.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_SRC)
    env.pop(ROOT_ENV_VAR_NAME, None)
    for name, value in env_overrides.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value

    result = subprocess.run(
        [sys.executable, "-c", body, argument],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"probe for {argument!r} failed (exit={result.returncode}):\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def _make_project(root: Path, marker: str = "pyproject.toml") -> Path:
    """A directory that looks like a project because it carries one marker, and nothing else."""
    root.mkdir(parents=True, exist_ok=True)
    (root / marker).write_text("")
    return root


def _nowhere(tmp_path: Path) -> Path:
    """A directory with no project marker above it, for the cases that must refuse."""
    outside = tmp_path / "nowhere" / "deeper"
    outside.mkdir(parents=True)
    return outside


class TestTheLibraryNeverInfersTheRootFromItsOwnLocation:
    """Ruling one: an installed copy must not be able to answer with a path inside site-packages."""

    def test_an_installed_copy_refuses_rather_than_answering_with_a_path_inside_the_environment(
        self, tmp_path: Path
    ) -> None:
        """Re-enact the measured trap: the library, installed, asked where the project is.

        The copy is placed at the exact depth a wheel lands at, so that the expression that bit this
        project -- ``parents[2]`` from the module's own file -- resolves to ``.../python3.13`` and
        would be returned without complaint. A test that only called ``project_root`` from inside the
        repository could never see this: in a repository, inferring from ``__file__`` gives the right
        answer, which is precisely why the bug survives review. The only honest measurement is from
        a copy that has been moved.
        """
        site_packages = tmp_path / "client" / ".venv" / "lib" / "python3.13" / "site-packages"
        site_packages.mkdir(parents=True)
        shutil.copytree(
            LIBRARY_SRC,
            site_packages / LIBRARY_SRC.name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        virtualenv_interior = site_packages.parent

        answer = _probe(_INSTALLED_COPY_PROBE, str(site_packages), cwd=_nowhere(tmp_path))

        # Guard the geometry before judging the answer: if the interpreter loaded the repository's
        # own copy instead of the moved one, this test would pass for the wrong reason forever.
        loaded = Path(answer["module_file"]).resolve()
        assert loaded.is_relative_to(site_packages), (
            f"the probe imported {loaded}, not the copy under {site_packages}; "
            f"the installed-copy geometry was never exercised"
        )
        assert loaded.parents[2] == virtualenv_interior, (
            f"the copy is not at wheel depth: parents[2] of {loaded} is {loaded.parents[2]}, "
            f"not {virtualenv_interior}, so the measured failure is not being reproduced"
        )

        assert "answered" not in answer, (
            f"an installed copy answered {answer['answered']!r} when asked where the project is. "
            f"Nothing above it is a project, so the only correct outcome is a refusal; a path here "
            f"is the silent failure this module exists to prevent."
        )
        assert ROOT_ENV_VAR_NAME in answer["refused"], (
            f"the refusal {answer['refused']!r} does not name {ROOT_ENV_VAR_NAME}, which is the one "
            f"thing a consumer in this position can do about it"
        )

    def test_importing_the_module_resolves_no_root_of_its_own(self, tmp_path: Path) -> None:
        """Import must be inert: no module-level constant may hold a root, and nothing may raise.

        A constant computed at import is the trap wearing a different hat -- it answers once, for
        everybody, before any caller has said where they are. The property checked is the shape of
        that mistake rather than the name of any constant: no module-level ``Path`` may be a
        directory that contains the module's own file. ``Path(__file__).parents[2]`` is caught by
        that whether it is called ``PROJECT_ROOT``, ``ROOT`` or anything else.

        Importing happens in a directory with no project above it, which also pins the second half:
        a library that refused at import time would be unimportable in exactly the environments it
        is meant to be installed into.
        """
        probe = _probe(_LIBRARY_IMPORT_PROBE, str(REPO_SRC), cwd=_nowhere(tmp_path))

        assert not probe["inferred"], (
            f"importing the module resolved {probe['inferred']} from its own location. Each of "
            f"those is a directory containing {probe['module_file']}, so it was derived from "
            f"__file__ -- the expression that returns the virtualenv once the library is installed."
        )

    def test_the_caller_must_say_where_it_is(self) -> None:
        """No argument, no answer: the start has no default, because the library cannot guess one.

        The other way to reintroduce the defect is a default -- ``project_root()`` falling back to
        the cwd, or to the library's own file. Either reads as convenience and behaves as a wrong
        answer for anyone whose process was launched from somewhere else.
        """
        with pytest.raises(TypeError):
            project_root()  # type: ignore[call-arg]


class TestTheEnvironmentVariableOverrides:
    """Ruling two: an explicit root wins, under a name the library owns.

    Every real run in this repository is a worktree against one editable install, and the override
    is how a run is told which checkout it belongs to. The library needs the same escape hatch under
    its own name, because it cannot read the client's.
    """

    def test_the_variable_publishes_the_name_the_scripts_and_containers_will_spell(self) -> None:
        """The name is the contract: a container sets it, CI sets it, and no code reads the other one."""
        assert ROOT_ENV_VAR == ROOT_ENV_VAR_NAME

    def test_the_variable_wins_over_the_directory_the_caller_started_from(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A caller sitting inside one project, told the root is another, gets the one it was told.

        The start deliberately sits inside a perfectly good project of its own, so that an
        implementation consulting the variable only as a fallback -- when the search found nothing --
        fails here. That ordering is the whole point: the override exists for the case where the
        search would otherwise succeed with the wrong answer.
        """
        chosen = _make_project(tmp_path / "chosen")
        other = _make_project(tmp_path / "other")
        start = other / "pkg" / "module.py"
        start.parent.mkdir(parents=True)
        start.write_text("")
        monkeypatch.setenv(ROOT_ENV_VAR_NAME, str(chosen))

        assert project_root(start) == chosen

    @pytest.mark.parametrize("broken", ("missing", "unmarked"))
    def test_a_variable_that_does_not_name_a_project_refuses_by_name(
        self, broken: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An override pointing at nothing, or at a directory that is not a project, must refuse.

        Both sides of the same rule. A typo in a deployment variable is the likeliest way this is
        ever wrong in production, and trusting the override because somebody set it deliberately is
        how ``ACME_PROJECT_ROOT`` pointing at a stale checkout goes unnoticed: every subsequent read
        finds a directory that exists and files that are simply the wrong ones. Refusing here costs a
        restart; accepting costs a rebuild from the wrong source.
        """
        target = tmp_path / "gone" if broken == "missing" else tmp_path / "unmarked"
        if broken == "unmarked":
            target.mkdir()
        monkeypatch.setenv(ROOT_ENV_VAR_NAME, str(target))
        start = _make_project(tmp_path / "real") / "module.py"
        start.write_text("")

        with pytest.raises(ProjectRootNotFound) as caught:
            project_root(start)

        message = str(caught.value)
        assert ROOT_ENV_VAR_NAME in message and str(target) in message, (
            f"the refusal {message!r} names neither the variable nor the value it was given, so the "
            f"reader cannot tell which of the two roots in play was rejected"
        )


class TestTheUpwardSearch:
    """Ruling two's other half: with no override, the root is found from where the caller says it is."""

    @pytest.mark.parametrize("marker", ROOT_MARKERS)
    @pytest.mark.parametrize("start_is_a_file", (True, False), ids=("from-a-file", "from-a-directory"))
    def test_every_published_marker_anchors_a_root(self, marker: str, start_is_a_file: bool, tmp_path: Path) -> None:
        """Each marker the module publishes must actually anchor a search, from a file or a directory.

        Parametrised over ``ROOT_MARKERS`` rather than over a list written here, so a marker added to
        the published tuple and forgotten in the search is a failure rather than documentation. The
        markers are created as plain files on purpose: a linked git worktree's ``.git`` is a file
        containing ``gitdir:``, not a directory, and this repository is worked on exclusively in
        worktrees -- an implementation testing ``is_dir()`` would refuse to find the very checkout
        this test is running in.

        Both start kinds are pinned because callers will pass both: a script passes ``__file__``, and
        a caller that has already computed a directory passes that.
        """
        root = _make_project(tmp_path / "project", marker=marker)
        nested = root / "src" / "package"
        nested.mkdir(parents=True)
        start = nested / "module.py" if start_is_a_file else nested
        if start_is_a_file:
            start.write_text("")

        found = project_root(start)

        assert found == root
        assert found.is_absolute(), f"{found} is not absolute; every caller joins other paths onto it"

    def test_a_git_directory_anchors_a_root_just_as_a_git_file_does(self, tmp_path: Path) -> None:
        """The ordinary clone, pinned as the mirror of the worktree case above.

        The two halves would otherwise be one test's worth of evidence for a rule with two shapes:
        fix the worktree case with ``is_file()`` and every ordinary clone stops resolving, with
        nothing red to say so.
        """
        root = tmp_path / "clone"
        (root / ".git" / "objects").mkdir(parents=True)
        start = root / "scripts" / "page.py"
        start.parent.mkdir(parents=True)
        start.write_text("")

        assert project_root(start) == root

    def test_the_nearest_project_wins_when_one_sits_inside_another(self, tmp_path: Path) -> None:
        """Search upward and stop, rather than walking to the outermost marker.

        This is not hypothetical here: a checkout containing a vendored library, or a worktree
        directory living inside the repository it was cut from, produces exactly this nesting. The
        outer answer is always plausible and always wrong -- it is the borrowed-checkout failure in
        miniature.
        """
        outer = _make_project(tmp_path / "outer")
        inner = _make_project(outer / "vendor" / "inner")
        start = inner / "src" / "module.py"
        start.parent.mkdir(parents=True)
        start.write_text("")

        assert project_root(start) == inner


class TestARefusalNamesWhatItLookedForAndWhereItLooked:
    """Ruling three: a wrong answer must be loud, and a refusal must be actionable.

    The value of this module is concentrated in this one message. Somebody reads it inside a
    container, at the bottom of a CI log, or in a client repository they have never opened, and the
    three facts they need are: what counts as a project, where the library searched, and how to say
    it explicitly. A refusal reading "could not determine project root" sends them to the source.
    """

    def test_the_refusal_names_the_start_the_markers_and_the_escape_hatch(self, tmp_path: Path) -> None:
        """No marker anywhere above the start: refuse by a named type, and say all three things."""
        start = _nowhere(tmp_path) / "module.py"
        start.write_text("")

        with pytest.raises(ProjectRootNotFound) as caught:
            project_root(start)

        message = str(caught.value)
        assert str(start.parent) in message, f"the refusal {message!r} does not say where it started looking"
        missing = [marker for marker in ROOT_MARKERS if marker not in message]
        assert not missing, f"the refusal {message!r} does not name marker(s) {missing}, so it cannot be acted on"
        assert ROOT_ENV_VAR_NAME in message, f"the refusal {message!r} does not name the override that would fix it"


class TestTheViewerScriptsNoLongerReachIntoClientCode:
    """Ruling four: the four scripts ship with the library and must behave identically without it."""

    @pytest.mark.parametrize("script_name", LIBRARY_SCRIPTS)
    def test_importing_a_viewer_script_pulls_in_no_client_code(self, script_name: str) -> None:
        """Measured by executing the script in a fresh interpreter and reading ``sys.modules``.

        Not by searching the source for the word: a grep is satisfied by a comment, defeated by an
        alias, and blind to the import arriving through some other module the script pulls in. What
        matters is whether the client package is in the process afterwards, which is the only form
        of the question that survives the library being installed somewhere that has no client
        package to import at all.
        """
        probe = _probe(_SCRIPT_PROBE, str(SCRIPTS_DIR / script_name), cwd=REPO_ROOT)

        assert CLIENT_PACKAGE not in probe["top_level"], (
            f"{script_name} imported {CLIENT_PACKAGE!r}. The scripts ship with the library, so the "
            f"client package will not exist wherever they land next."
        )

    @pytest.mark.parametrize("script_name", LIBRARY_SCRIPTS)
    def test_every_path_a_viewer_script_resolves_lands_inside_this_checkout(self, script_name: str) -> None:
        """Run from this repository, a script must address this repository and nothing else.

        The one property that catches the failure that has actually happened, and it catches it for
        every path a script computes rather than for the ones named below. A root resolved from an
        editable install pointing at another checkout, or from inside a virtualenv, shows up here as
        a path that is not under this worktree -- which is what "a page built from a pipeline file
        the author had never touched" looked like from the outside.
        """
        probe = _probe(_SCRIPT_PROBE, str(SCRIPTS_DIR / script_name), cwd=REPO_ROOT)

        strays = {name: value for name, value in probe["paths"].items() if not Path(value).is_relative_to(REPO_ROOT)}
        assert not strays, (
            f"{script_name} resolved {strays} outside {REPO_ROOT}. Run from this checkout it must "
            f"address this checkout; a path elsewhere is the borrowed-checkout defect."
        )

    @pytest.mark.parametrize(
        ("script_name", "expected"),
        (
            ("serve_runs.py", ("", "output_files/runs")),
            (
                "tool_catalog_page.py",
                ("output_files", "output_files/tool_catalog.html", "output_files/tool_catalog.json"),
            ),
            ("etl_architecture_page.py", ("output_files/runs/architecture",)),
        ),
    )
    def test_a_viewer_script_still_addresses_the_same_places_as_before(
        self, script_name: str, expected: tuple[str, ...]
    ) -> None:
        """Identical behaviour, stated as the set of places each script reads and writes.

        Checked by value rather than by constant name, so renaming ``RUNS_ROOT`` is free and moving
        where runs are served from is not -- which is the right way round, because a reader's
        bookmarked URL and every run already on disk depend on the location, not the spelling.
        ``serve_runs`` is expected to hold the checkout root itself: that is what it hands to the
        preview service as ``source_root``, and it is what makes "show me the code behind this step"
        answer from this repository.

        ``pipeline_walkthrough_page.py`` is covered by the containment property above instead; it is
        under active change, and pinning a filename it is in the middle of moving would be a test
        about today's bytes rather than about the root.
        """
        probe = _probe(_SCRIPT_PROBE, str(SCRIPTS_DIR / script_name), cwd=REPO_ROOT)

        resolved = {Path(value) for value in probe["paths"].values()}
        wanted = {REPO_ROOT / relative if relative else REPO_ROOT for relative in expected}
        assert wanted <= resolved, (
            f"{script_name} no longer addresses {sorted(str(p) for p in wanted - resolved)}; "
            f"it resolved {sorted(str(p) for p in resolved)}"
        )

    def test_the_environment_variable_still_moves_a_scripts_root(self, tmp_path: Path) -> None:
        """The worktree workflow, preserved: point a script at a root and everything it touches moves.

        ``serve_runs`` carries the least machinery of the four, so it measures the root and nothing
        else. If the override stopped reaching the scripts, every worktree run would silently serve
        the main checkout's runs -- which is the failure this whole module is a response to, with the
        client's variable name filed off.
        """
        elsewhere = _make_project(tmp_path / "elsewhere")

        probe = _probe(
            _SCRIPT_PROBE,
            str(SCRIPTS_DIR / "serve_runs.py"),
            cwd=REPO_ROOT,
            env_overrides={ROOT_ENV_VAR_NAME: str(elsewhere)},
        )

        resolved = {Path(value) for value in probe["paths"].values()}
        assert elsewhere / "output_files" / "runs" in resolved, (
            f"with {ROOT_ENV_VAR_NAME}={elsewhere} the script resolved {sorted(str(p) for p in resolved)}; "
            f"the override did not reach it"
        )
        assert not any(path.is_relative_to(REPO_ROOT) for path in resolved), (
            f"the override moved some paths but not all: {sorted(str(p) for p in resolved)} still "
            f"contains paths under {REPO_ROOT}, so a run would read one root and write another"
        )
