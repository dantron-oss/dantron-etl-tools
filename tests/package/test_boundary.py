"""Boundary tests for the dantron_etl package.

dantron_etl is a generic, client-neutral library. It must never depend on client code, an
orchestrator (dagster, prefect) or pandas, because whatever it depends on ships to every consumer.
These tests enforce that, and keep client vocabulary out of the package's source and shipped docs.
"""

import ast
import json
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

# Imports each module named on the command line, in order, in one fresh interpreter. After each one
# it notes any forbidden top-level package that has just appeared, so a failure names the module
# that first pulled it in.
_IMPORT_CHECK_SCRIPT = """
import importlib, json, sys

forbidden = set(sys.argv[1].split(","))
seen = set()
blame = {}
for name in sys.argv[2:]:
    importlib.import_module(name)
    arrived = {module.split(".")[0] for module in sys.modules} & forbidden
    for package in sorted(arrived - seen):
        blame[package] = name
    seen |= arrived
print(json.dumps(blame))
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


def _forbidden_packages_pulled_in(targets: list[str], forbidden: frozenset[str] = FORBIDDEN_IMPORTS) -> dict[str, str]:
    """Import every target in one fresh interpreter; map each forbidden package to what pulled it in.

    A fresh interpreter, because this pytest session has imported plenty already, so checking its
    own ``sys.modules`` would prove nothing. One interpreter for all of them, because importing any
    submodule runs ``dantron_etl/__init__``, which imports every tool: a process per module paid
    the same start-up cost forty times to get the same answer.
    """
    env = os.environ.copy()
    src_dir = str(SRC_DIR)
    existing_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{existing_path}" if existing_path else src_dir

    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK_SCRIPT, ",".join(sorted(forbidden)), *targets],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Importing {targets} in a fresh subprocess failed "
        f"(exit={result.returncode}):\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestDantronEtlImportBoundary:
    """No dantron_etl module, the viewer and the command included, pulls in client code."""

    def test_no_module_imports_a_forbidden_package(self) -> None:
        """Importing every module must not load acme/dagster/prefect/pandas.

        This measures what actually lands in ``sys.modules``, so it also sees an import that arrives
        through some third module. The source scan below sees imports inside functions, which this
        cannot.
        """
        blame = _forbidden_packages_pulled_in(_discover_modules())
        assert not blame, f"forbidden package(s) were imported; each is named with what pulled it in: {blame}"

    def test_the_check_names_the_module_that_pulled_a_forbidden_package_in(self) -> None:
        """The check above can fail: with ``colorsys`` standing in for a forbidden package, it is caught.

        ``colorsys`` is in the standard library, so it is always there, and nothing in the library
        imports it. Without this, a probe that never reported anything would pass for ever.
        """
        blame = _forbidden_packages_pulled_in(["dantron_etl", "colorsys"], forbidden=frozenset({"colorsys"}))
        assert blame == {"colorsys": "colorsys"}

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
        from dantron_etl.authoring import REGISTRY

        text = self._manual()
        missing = sorted(tool.name for tool in REGISTRY.tools() if f"`{tool.name}`" not in text)
        assert not missing, f"MANUAL.md names no tool called {missing}; its catalog table is stale"

    def test_the_manual_names_no_tool_that_does_not_exist(self) -> None:
        """The failure that matters: a reader writes the name they were given and the run refuses it."""
        import dantron_etl  # noqa: F401
        from dantron_etl.authoring import REGISTRY

        registered = {tool.name for tool in REGISTRY.tools()}
        table = self._manual().split("### 3.1 The tools that ship", 1)[-1].split("---", 1)[0]
        named = {match.strip("`") for match in re.findall(r"`[a-z_]+`", table)}
        invented = sorted(named - registered)
        assert not invented, f"MANUAL.md names {invented}, which no tool provides"

    def test_the_stated_tool_count_is_the_real_one(self) -> None:
        """The number a reader weighs 'is something missing?' against."""
        import dantron_etl  # noqa: F401
        from dantron_etl.authoring import REGISTRY

        count = len(list(REGISTRY.tools()))
        assert f"{count} tools" in self._manual(), f"MANUAL.md does not say '{count} tools'; the registry holds {count}"

    def test_every_category_the_contract_publishes_is_listed(self) -> None:
        """A contributor picks a category from this list, and one it omits is one nobody declares."""
        from dantron_etl.authoring import CATEGORIES

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
# The library's own tests travel with the package. The check below scans their source for a
# forbidden import at any depth, including inside a function. The viewer and the command live
# inside the package, so the checks above already cover them.
#
# An earlier version also imported every test file in a subprocess with the client package blocked,
# one subprocess per file. That guarded the move out of the client repository. The source scan of
# tests/ below catches the same import directly, at any depth, whether or not a client package
# happens to be installed; anything a test imports through the library is covered by the import
# check above.
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parents[1]


class TestWhatShipsBesideThePackageSurvivesExtraction:
    """The tests that travel with the library are free of client imports."""

    @pytest.mark.parametrize(
        "test_path", sorted(TESTS_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(TESTS_DIR))
    )
    def test_no_forbidden_import_appears_anywhere_in_a_test_file(self, test_path: Path) -> None:
        """A test that imports client code, an orchestrator or pandas cannot run where the library ships."""
        offenders = sorted(_imported_roots(test_path) & FORBIDDEN_IMPORTS)
        assert not offenders, f"tests/{test_path.relative_to(TESTS_DIR)} imports forbidden package(s) {offenders}"


class TestShippedTestsNameNoClientModules:
    """A module name a shipped test invents is an identifier it creates, and it travels as source.

    Narrow on purpose. The vocabulary fence over the package's own source cannot be pointed at test
    files: they are full of ordinary English -- a test that reconciles two frames says so -- and
    sample data has to be about something. What is checked here is not prose but the name a test
    registers in ``sys.modules`` when it loads a file by path, which today carries this
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

    @pytest.mark.parametrize("test_path", sorted(TESTS_DIR.rglob("test_*.py")), ids=lambda p: p.name)
    def test_no_module_it_loads_by_path_is_named_after_the_client(self, test_path: Path) -> None:
        """The name is invented at the call site, so a client's name in it is a choice, not a constraint."""
        # The library's own name is exempt for the reason a manual may print it: a test loading a
        # file the library ships may say so. Every other term on the list names somebody's client.
        found = {
            name: [term for term in DOC_FORBIDDEN_TERMS if term in name.lower()]
            for name in self._module_names_loaded_by_path(test_path)
        }
        named = {name: terms for name, terms in found.items() if terms}
        assert not named, f"tests/{test_path.relative_to(TESTS_DIR)} loads module(s) named {named}, which name a client"
