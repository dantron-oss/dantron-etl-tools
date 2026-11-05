"""Boundary tests for the dantron_etl package.

dantron_etl is a generic, client-agnostic transformation library that will eventually ship as its
own PyPI distribution and be reused by clients other than this repo's acme package. The one rule
that governs it: it must never depend on client code (acme), any orchestrator (dagster, prefect),
or heavy client-shaped dependencies (pandas) -- because that dependency would ship to every other
consumer of the library. These tests enforce that boundary automatically as modules are added, and
guard against client-specific vocabulary leaking into the package's source, docstrings, or comments.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dantron_etl.paths import project_root

PROJECT_ROOT = project_root(__file__)
SRC_DIR = PROJECT_ROOT / "src"
DANTRON_ETL_SRC = SRC_DIR / "dantron_etl"

FORBIDDEN_IMPORTS = frozenset({"acme", "dagster", "prefect", "pandas"})

FORBIDDEN_TERMS = (
    "ledger",
    "crm",
    "acme",
    "dantron",
    "rwd",
    "mrr",
    "invoice",
    "tenant",
    "reconcil",
)

_IMPORT_CHECK_SCRIPT = """
import importlib
import sys

importlib.import_module("{module}")
top_level = {{name.split(".")[0] for name in sys.modules}}
print(",".join(sorted(top_level)))
"""


def _discover_source_files() -> list[Path]:
    """Every .py file under src/dantron_etl, discovered by walking the filesystem."""
    if not DANTRON_ETL_SRC.is_dir():
        return []
    return sorted(DANTRON_ETL_SRC.rglob("*.py"))


def _discover_modules() -> list[str]:
    """Dotted module names for every .py file under src/dantron_etl.

    Discovered by walking the filesystem, not hardcoded, so modules added in later work are
    automatically covered by this same test.
    """
    modules: list[str] = []
    for path in _discover_source_files():
        relative = path.relative_to(SRC_DIR)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if not parts:
            continue
        modules.append(".".join(parts))
    return modules


def _imported_roots(source_path: Path) -> set[str]:
    """Every top-level package name imported anywhere in the file, at any nesting depth."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source_path.read_text(encoding="utf8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_top_level_packages(module_name: str) -> set[str]:
    """Top-level packages visible in sys.modules inside a fresh subprocess that imports module_name.

    This must run in a fresh subprocess rather than in-process: the pytest session running this
    test has already imported acme (via conftest.py and other test modules), so an in-process
    sys.modules check would pass vacuously.
    """
    env = os.environ.copy()
    src_dir = str(SRC_DIR)
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_path}" if existing_path else src_dir

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK_SCRIPT.format(module=module_name)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"Importing {module_name!r} in a fresh subprocess failed "
        f"(exit={result.returncode}):\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    output_lines = result.stdout.strip().splitlines()
    last_line = output_lines[-1] if output_lines else ""
    return set(last_line.split(",")) if last_line else set()


class TestDantronEtlImportBoundary:
    """Every dantron_etl module must be importable without pulling in client code."""

    @pytest.mark.parametrize("module_name", _discover_modules())
    def test_module_import_has_no_client_dependencies(self, module_name: str) -> None:
        """Importing a dantron_etl module in isolation must not import acme/dagster/prefect/pandas."""
        imported = _imported_top_level_packages(module_name)
        offending = imported & FORBIDDEN_IMPORTS
        assert not offending, (
            f"Module {module_name!r} pulled in forbidden package(s) {sorted(offending)} via its "
            f"import graph (observed sys.modules top-level packages: {sorted(imported)})"
        )

    def test_at_least_one_module_is_discovered(self) -> None:
        """Guard against the discovery walk silently finding zero modules once the package exists."""
        assert _discover_modules(), f"No .py modules found under {DANTRON_ETL_SRC}"


class TestDantronEtlSourceVocabulary:
    """dantron_etl source text (including docstrings/comments) must stay client-agnostic."""

    @pytest.mark.parametrize("source_path", _discover_source_files(), ids=lambda p: p.name)
    def test_source_file_has_no_forbidden_terms(self, source_path: Path) -> None:
        """No forbidden client/product term may appear anywhere in a dantron_etl source file."""
        text = source_path.read_text().lower()
        matched = [term for term in FORBIDDEN_TERMS if term in text]
        assert not matched, f"{source_path} contains forbidden term(s): {matched}"


class TestDantronEtlVersion:
    """dantron_etl must expose a usable package version ahead of its eventual PyPI extraction."""

    def test_dunder_version_is_a_non_empty_string(self) -> None:
        """dantron_etl.__version__ must exist and be a non-empty string."""
        import dantron_etl

        assert isinstance(dantron_etl.__version__, str)
        assert dantron_etl.__version__.strip() != ""


class TestDantronEtlLazyImports:
    """A forbidden import must be caught wherever it sits, not only at module scope.

    The subprocess check above sees what importing a module actually pulls in, which is the honest
    test of a module's cost -- but it is blind to an import written inside a function body, because
    that import never runs. That is not a hypothetical gap: deferring an import into a function is
    exactly the trick used elsewhere in this repo to keep an orchestrator out of a lightweight
    process, so it is the most likely way this boundary would ever be crossed. Parsing the source
    catches it at any nesting depth, at the cost of not knowing whether the branch is reachable --
    which is the right trade, because a forbidden import that is merely hard to reach is still a
    dependency this library must not ship.
    """

    @pytest.mark.parametrize("source_path", _discover_source_files(), ids=lambda p: p.name)
    def test_no_forbidden_import_appears_anywhere_in_the_source(self, source_path: Path) -> None:
        """No forbidden package may be imported at module scope, in a function, or in a branch."""
        offenders = sorted(_imported_roots(source_path) & FORBIDDEN_IMPORTS)
        assert not offenders, (
            f"{source_path} imports forbidden package(s) {offenders} somewhere in its source. "
            f"An import inside a function still makes this a dependency the library would ship."
        )


# The library's own distribution name. It is the one term on the forbidden list that a document
# shipped INSIDE the package cannot avoid -- a manual has to name the thing it documents, and the
# import lines it shows a reader are exactly the lines that spell it. It stays forbidden in .py,
# where relative imports mean it has never once been needed.
PACKAGE_NAME = "dantron"

DOC_FORBIDDEN_TERMS = tuple(term for term in FORBIDDEN_TERMS if term != PACKAGE_NAME)


def _discover_doc_files() -> list[Path]:
    """Every .md file shipped inside the package, discovered by walking the filesystem."""
    if not DANTRON_ETL_SRC.is_dir():
        return []
    return sorted(DANTRON_ETL_SRC.rglob("*.md"))


class TestDantronEtlDocumentVocabulary:
    """A document shipped inside the package is source too, and leaks client vocabulary just as far.

    The checks above walk .py files only, which was right while the package held nothing else. A
    manual shipped beside the code is read by every consumer of the library and travels into every
    other client's repository with it, so it is held to the same rule -- minus the package's own
    name, which it must be able to write.
    """

    @pytest.mark.parametrize("doc_path", _discover_doc_files(), ids=lambda p: p.name)
    def test_document_has_no_forbidden_terms(self, doc_path: Path) -> None:
        """No client or product term may appear in a document the package ships."""
        text = doc_path.read_text().lower()
        matched = [term for term in DOC_FORBIDDEN_TERMS if term in text]
        assert not matched, f"{doc_path} contains forbidden term(s): {matched}"

    def test_the_manual_ships_inside_the_package(self) -> None:
        """The manual travels with the directory, or it is gone once the library is extracted."""
        assert (DANTRON_ETL_SRC / "MANUAL.md").is_file(), (
            f"No MANUAL.md under {DANTRON_ETL_SRC}. It lives inside the package on purpose: "
            f"extraction is meant to be a move of one directory with no renames."
        )


class TestTheManualDescribesTheLibraryThatExists:
    """A manual naming a tool nobody can call is worse than no manual.

    It is read by agents that write a pipeline file straight from it, so a stale name does not read
    as documentation drift -- it reads as a step that fails at load under a name the reader was told
    to use. The counts are checked for the same reason: a reader deciding whether to build a new
    tool is deciding against the list they were shown.
    """

    @staticmethod
    def _manual() -> str:
        return (DANTRON_ETL_SRC / "MANUAL.md").read_text()

    def test_every_registered_tool_is_named_in_the_manual(self) -> None:
        """A tool the manual omits is one no reader knows to reach for before building their own."""
        import dantron_etl  # noqa: F401  -- importing the package is what populates the registry
        from dantron_etl.tool import REGISTRY

        text = self._manual()
        missing = sorted(tool.name for tool in REGISTRY.tools() if f"`{tool.name}`" not in text)
        assert not missing, f"MANUAL.md names no tool called {missing}; its catalog table is stale"

    def test_the_manual_names_no_tool_that_does_not_exist(self) -> None:
        """The failure that matters: a reader writes the name they were given and the run refuses it."""
        import dantron_etl  # noqa: F401
        from dantron_etl.tool import REGISTRY

        registered = {tool.name for tool in REGISTRY.tools()}
        table = self._manual().split("### 3.1 The tools that ship", 1)[-1].split("---", 1)[0]
        named = {match.strip("`") for match in re.findall(r"`[a-z_]+`", table)}
        invented = sorted(named - registered)
        assert not invented, f"MANUAL.md names {invented}, which no tool provides"

    def test_the_stated_tool_count_is_the_real_one(self) -> None:
        """The number a reader weighs 'is something missing?' against."""
        import dantron_etl  # noqa: F401
        from dantron_etl.tool import REGISTRY

        count = len(list(REGISTRY.tools()))
        assert f"{count} tools" in self._manual(), f"MANUAL.md does not say '{count} tools'; the registry holds {count}"

    def test_every_category_the_contract_publishes_is_listed(self) -> None:
        """A contributor picks a category from this list, and one it omits is one nobody declares."""
        from dantron_etl.tool import CATEGORIES

        text = self._manual()
        missing = sorted(name for name in CATEGORIES if name not in text)
        assert not missing, f"MANUAL.md lists no category {missing}, which a new tool may declare"

    def test_every_whitelisted_expression_function_is_listed(self) -> None:
        """The list a reader checks before concluding the language cannot do something."""
        from dantron_etl.expr import ALLOWED_FUNCTIONS

        text = self._manual()
        missing = sorted(name for name in ALLOWED_FUNCTIONS if f"`{name}`" not in text)
        assert not missing, f"MANUAL.md omits whitelisted function(s) {missing}"


# ---------------------------------------------------------------------------
# What ships beside the package
#
# The fence above walks src/dantron_etl and stops there, which was right while the package was the
# only thing being extracted. It is no longer the whole shipment. Four viewer scripts under
# scripts/ are the library's user interface -- they serve runs, render the walkthrough, generate the
# catalog and draw the architecture -- and the library's own test files are the only evidence, on
# arrival in a new repository, that any of it works. Both travel with the directory; neither is
# fenced; each currently reaches into this client's package for a project root.
#
# The two failures are not equally visible, which is why the test files matter more. A script that
# cannot import is noticed the first time somebody runs it, and the person running it is holding the
# traceback. A test file that cannot import takes the whole suite down on the first `pytest` in the
# extracted repository -- the run whose entire purpose is to say whether the extraction worked.
#
# The property measured is arrival, not vocabulary: does this file import in an interpreter where
# the client package does not exist? That is the question the extracted repository asks, and it is
# the only form of the question that a comment, an alias, or an import arriving through some third
# module cannot slip past. It also, correctly, permits a guarded fallback: a shipped test that says
# "use the host project's database URL if there is a host project" inside a try/except has already
# been written to degrade, and the branch that runs after extraction is the one that returns None.
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parent / "scripts"

#: The viewer scripts that ship with the library. Listed rather than discovered, because there is no
#: marker on disk that separates them from the client's own scripts -- several of which import
#: dantron_etl legitimately and must go on importing client code, being client code themselves. A
#: list that a rename would rot is guarded by the existence test below.
LIBRARY_SCRIPTS = (
    "serve_runs.py",
    "pipeline_walkthrough_page.py",
    "tool_catalog_page.py",
    "etl_architecture_page.py",
)

#: The library's own tests, discovered by the naming convention every one of them already follows,
#: so the next one written is fenced by existing rather than by being remembered here.
LIBRARY_TEST_GLOB = "test_dantron_etl_*.py"

#: And the tests for the scripts that ship beside the library, which the glob cannot reach because
#: they are named after the script rather than after the package. They are listed rather than
#: renamed: a test file's name is what a person greps for, and moving it to satisfy a glob buys
#: nothing the list does not. The guard below refuses a name in here that is not on disk, so a
#: rename breaks loudly instead of quietly shrinking the fence.
SHIPPED_SCRIPT_TESTS: tuple[str, ...] = (
    "test_tool_catalog_page.py",
    "test_pipeline_walkthrough_page.py",
)

_ARRIVAL_SCRIPT = '''
import importlib.abc, importlib.util, sys

blocked = set(sys.argv[2].split(","))


class _Absent(importlib.abc.MetaPathFinder):
    """An interpreter in which the client's packages simply are not installed."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, _Absent())
for name in [n for n in sys.modules if n.split(".")[0] in blocked]:
    del sys.modules[name]

spec = importlib.util.spec_from_file_location("shipped_under_test", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print("imported")
'''


def _discover_library_test_files() -> list[Path]:
    """Every test file that belongs to the library rather than to the client.

    The glob plus the named tests for the shipped scripts. Both sets travel with the package when it
    is extracted, so both have to import in a world where this client's code does not exist -- and
    the tests for the scripts are the easier half to forget, being named after what they test rather
    than after the package they belong to.
    """
    named = [TESTS_DIR / name for name in SHIPPED_SCRIPT_TESTS]
    missing = [path.name for path in named if not path.is_file()]
    assert not missing, (
        f"SHIPPED_SCRIPT_TESTS names {missing}, which are not under {TESTS_DIR}; a renamed file "
        f"must be renamed here too, or the fence quietly stops covering it"
    )
    return sorted(set(TESTS_DIR.glob(LIBRARY_TEST_GLOB)) | set(named))


def _import_with_client_code_absent(path: Path) -> subprocess.CompletedProcess[str]:
    """Import one shipped file in a fresh interpreter where the forbidden packages do not exist.

    Blocking is done with a meta path finder rather than by building a stripped sys.path: the two
    source roots live side by side in one directory, so any PYTHONPATH that exposes dantron_etl from
    this checkout exposes the client package with it. Refusing at the finder reproduces the extracted
    repository exactly -- the package is not missing from the path, it does not exist.
    """
    env = os.environ.copy()
    src_dir = str(SRC_DIR)
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_path}" if existing_path else src_dir
    return subprocess.run(
        [sys.executable, "-c", _ARRIVAL_SCRIPT, str(path), ",".join(sorted(FORBIDDEN_IMPORTS))],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=120,
    )


class TestWhatShipsBesideThePackageSurvivesExtraction:
    """Everything that travels with the library must import where the client's code does not exist."""

    @pytest.mark.parametrize("script_name", LIBRARY_SCRIPTS)
    def test_a_viewer_script_imports_with_the_client_package_absent(self, script_name: str) -> None:
        """A script that needs client code to load is a script that stops working on extraction."""
        result = _import_with_client_code_absent(SCRIPTS_DIR / script_name)
        assert result.returncode == 0, (
            f"scripts/{script_name} could not be imported without {sorted(FORBIDDEN_IMPORTS)}:\n{result.stderr.strip()}"
        )

    @pytest.mark.parametrize("test_path", _discover_library_test_files(), ids=lambda p: p.name)
    def test_a_library_test_file_imports_with_the_client_package_absent(self, test_path: Path) -> None:
        """The suite is the only evidence the extracted package works, so it must run on arrival."""
        result = _import_with_client_code_absent(test_path)
        assert result.returncode == 0, (
            f"tests/{test_path.name} could not be imported without {sorted(FORBIDDEN_IMPORTS)}, so "
            f"collection fails and the whole suite is lost on the first run after extraction:\n"
            f"{result.stderr.strip()}"
        )

    def test_every_named_viewer_script_is_where_the_list_says(self) -> None:
        """A renamed script must fail loudly here rather than silently leaving the fence."""
        missing = [name for name in LIBRARY_SCRIPTS if not (SCRIPTS_DIR / name).is_file()]
        assert not missing, f"{missing} named in LIBRARY_SCRIPTS but not found under {SCRIPTS_DIR}"

    def test_the_library_tests_are_discovered(self) -> None:
        """Guard against the glob quietly matching nothing and the fence passing over an empty set."""
        assert _discover_library_test_files(), f"no {LIBRARY_TEST_GLOB} under {TESTS_DIR}"


class TestLibraryScriptsHaveNoDeferredClientImports:
    """A viewer script's forbidden import counts wherever it sits, unlike the class above.

    The arrival check sees only what runs at import, and a script's work happens in ``main`` -- which
    is precisely where a deferred import would sit, and precisely when it would fail: not on the
    first `pytest` in the extracted repository, but the first time a person tries to look at a run.
    Parsing catches it at any depth.

    The library's test files are deliberately held to the looser rule instead. A guarded import in a
    shipped test is a documented degradation with a branch that runs after extraction; the same line
    in a script is a script with a hole in it.
    """

    @pytest.mark.parametrize("script_name", LIBRARY_SCRIPTS)
    def test_no_forbidden_import_appears_anywhere_in_a_viewer_script(self, script_name: str) -> None:
        """No forbidden package may be imported at module scope, in a function, or in a branch."""
        offenders = sorted(_imported_roots(SCRIPTS_DIR / script_name) & FORBIDDEN_IMPORTS)
        assert not offenders, (
            f"scripts/{script_name} imports forbidden package(s) {offenders}. Deferred into a "
            f"function it still fails, later and in front of a reader rather than in the suite."
        )


class TestShippedTestsNameNoClientModules:
    """A module name a shipped test invents is an identifier it creates, and it travels as source.

    Narrow on purpose. The vocabulary fence over the package's own source cannot be pointed at test
    files: they are full of ordinary English -- a test that reconciles two frames says so -- and
    sample data has to be about something. What is checked here is not prose but the name a test
    registers in ``sys.modules`` when it loads a shipping script by path, which today carries this
    client's name in a file destined for a repository where no such thing exists. Nothing imports
    those names -- they are keys chosen at the call site -- so the rename is free, and the only cost
    of leaving them is a reader of the extracted library being told it belongs to a client they have
    never heard of.
    """

    @staticmethod
    def _module_names_loaded_by_path(source_path: Path) -> list[str]:
        """Every literal module name the file passes to ``spec_from_file_location``."""
        names: list[str] = []
        for node in ast.walk(ast.parse(source_path.read_text(encoding="utf8"))):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if called != "spec_from_file_location" or not node.args:
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                names.append(first.value)
        return names

    @pytest.mark.parametrize("test_path", _discover_library_test_files(), ids=lambda p: p.name)
    def test_no_module_it_loads_by_path_is_named_after_the_client(self, test_path: Path) -> None:
        """The name is invented at the call site, so a client's name in it is a choice, not a constraint."""
        # The library's own name is exempt for the reason a manual may print it: a test loading a
        # file the library ships may say so. Every other term on the list names somebody's client.
        found = {
            name: [term for term in DOC_FORBIDDEN_TERMS if term in name.lower()]
            for name in self._module_names_loaded_by_path(test_path)
        }
        named = {name: terms for name, terms in found.items() if terms}
        assert not named, f"tests/{test_path.name} loads module(s) named {named}, which name a client"
