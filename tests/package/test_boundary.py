"""The library stays generic: it imports nothing it must not, and no file names a client.

Two kinds of check live here.

- Imports. No module may import an orchestrator, pandas, or a client's own package. The generic
  names are listed below. A client's package name is a client word, so it comes from the local list.
- Words. No tracked file may hold a word that identifies a client or its systems. Those words are
  not written in this repository. Each maintainer keeps them in a local file, ``.forbidden-terms``
  at the repository root (git ignores it), one word per line, ``#`` for comments. The environment
  variable ``ETL_FORBIDDEN_TERMS`` (comma-separated) adds more, for CI. With neither, the word
  checks are skipped and say why.

A word matches anywhere, ignoring case, so ``acme`` also catches ``AcmeLedger``. A word that could
be a Python package name is also refused as an import.

The library's own name has one rule of its own: it never appears in a ``.py`` file under ``src/``,
which is why the package imports itself relatively.
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
PACKAGE_SRC = SRC_DIR / "dantron_etl"
TESTS_DIR = PROJECT_ROOT / "tests"

TERMS_FILE = PROJECT_ROOT / ".forbidden-terms"
TERMS_VARIABLE = "ETL_FORBIDDEN_TERMS"

# Packages no consumer should be made to carry.
GENERIC_FORBIDDEN_IMPORTS = frozenset({"dagster", "prefect", "pandas"})

# The library's own name. Allowed in documents, which must name what they describe; never in its .py.
PACKAGE_NAME = "dantron"

# Places a tracked-file walk skips when git is not there to list the files.
_NOT_SOURCE = {".git", ".venv", "venv", "node_modules", "output_files", "dist", "build", "__pycache__"}


def read_terms(file: Path = TERMS_FILE, variable: str = TERMS_VARIABLE) -> tuple[str, ...]:
    """The client words from the local file and the environment variable, lower-cased, in order."""
    words: list[str] = []
    if file.is_file():
        for line in file.read_text(encoding="utf8").splitlines():
            word = line.split("#", 1)[0].strip()
            if word:
                words.append(word)
    words.extend(part.strip() for part in os.environ.get(variable, "").split(","))
    return tuple(dict.fromkeys(word.lower() for word in words if word))


CLIENT_TERMS = read_terms()
NO_TERMS_REASON = (
    f"no client words to check: list them in {TERMS_FILE.name} at the repository root "
    f"(one per line) or in {TERMS_VARIABLE} (comma-separated). See CONTRIBUTING.md."
)
needs_terms = pytest.mark.skipif(not CLIENT_TERMS, reason=NO_TERMS_REASON)

FORBIDDEN_IMPORTS = GENERIC_FORBIDDEN_IMPORTS | {word for word in CLIENT_TERMS if word.isidentifier()}

# Imports each module named on the command line, in one fresh interpreter. After each one it notes
# any forbidden top-level package that has just appeared, so a failure names the module to blame.
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


def _source_files() -> list[Path]:
    """Every .py file under src/dantron_etl."""
    return sorted(PACKAGE_SRC.rglob("*.py")) if PACKAGE_SRC.is_dir() else []


def _modules() -> list[str]:
    """The dotted module name of every .py file under src/dantron_etl."""
    modules: list[str] = []
    for path in _source_files():
        parts = list(path.relative_to(SRC_DIR).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append(".".join(parts))
    return modules


def tracked_files(root: Path = PROJECT_ROOT) -> list[Path]:
    """Every file git tracks under ``root``. Without git (an unpacked sdist), every file not ignored here."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True, timeout=60
        ).stdout.decode("utf8")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        listed = ""
    if listed:
        return sorted(root / name for name in listed.split("\0") if name and (root / name).is_file())
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name != TERMS_FILE.name
        and not _NOT_SOURCE.intersection(path.relative_to(root).parts)
    )


def _text(path: Path) -> str | None:
    """The file's text, or None for a binary file such as a workbook."""
    try:
        return path.read_bytes().decode("utf8")
    except UnicodeDecodeError:
        return None


def _imported_roots(source_path: Path) -> set[str]:
    """Every top-level package the file imports, at any depth, including inside a function."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source_path.read_text(encoding="utf8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _forbidden_packages_pulled_in(targets: list[str], forbidden: frozenset[str]) -> dict[str, str]:
    """Import every target in one fresh interpreter. Map each forbidden package that arrived to its importer.

    A fresh interpreter, because this pytest session has imported plenty already.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK_SCRIPT, ",".join(sorted(forbidden)), *targets],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, f"importing {targets} failed:\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestImports:
    """No module of the library, the viewer and the command included, pulls in a forbidden package."""

    def test_no_module_imports_a_forbidden_package(self) -> None:
        """Measured by importing every module, so an import that arrives through a third module is seen too."""
        blame = _forbidden_packages_pulled_in(_modules(), FORBIDDEN_IMPORTS)
        assert not blame, f"forbidden package(s) imported, each with the module that pulled it in: {blame}"

    def test_the_check_names_the_module_that_pulled_a_forbidden_package_in(self) -> None:
        """The check can fail: ``colorsys`` stands in for a forbidden package and is caught."""
        blame = _forbidden_packages_pulled_in(["dantron_etl", "colorsys"], frozenset({"colorsys"}))
        assert blame == {"colorsys": "colorsys"}

    def test_modules_are_found(self) -> None:
        assert _modules(), f"no .py modules found under {PACKAGE_SRC}"

    @pytest.mark.parametrize("source_path", _source_files(), ids=lambda p: p.name)
    def test_no_forbidden_import_appears_anywhere_in_the_source(self, source_path: Path) -> None:
        """An import inside a function never runs in the check above, but it is still a dependency."""
        offenders = sorted(_imported_roots(source_path) & FORBIDDEN_IMPORTS)
        assert not offenders, f"{source_path} imports forbidden package(s) {offenders}"

    @pytest.mark.parametrize("test_path", sorted(TESTS_DIR.rglob("*.py")), ids=lambda p: str(p.relative_to(TESTS_DIR)))
    def test_no_forbidden_import_appears_in_a_test_file(self, test_path: Path) -> None:
        """The tests ship in the source package, so they must run where no client package exists."""
        offenders = sorted(_imported_roots(test_path) & FORBIDDEN_IMPORTS)
        assert not offenders, f"tests/{test_path.relative_to(TESTS_DIR)} imports forbidden package(s) {offenders}"


class TestNoFileNamesAClient:
    """No tracked file holds a client word: code, tests, docs, workflows and top-level files alike."""

    @needs_terms
    def test_no_tracked_file_holds_a_client_word(self) -> None:
        found: dict[str, list[str]] = {}
        for path in tracked_files():
            text = _text(path)
            if text is None:
                continue
            lowered = text.lower()
            matched = [word for word in CLIENT_TERMS if word in lowered]
            if matched:
                found[str(path.relative_to(PROJECT_ROOT))] = matched
        assert not found, f"files that name a client, with the words found: {found}"

    @needs_terms
    def test_no_file_or_folder_is_named_with_a_client_word(self) -> None:
        named = [
            str(path.relative_to(PROJECT_ROOT))
            for path in tracked_files()
            if any(word in str(path.relative_to(PROJECT_ROOT)).lower() for word in CLIENT_TERMS)
        ]
        assert not named, f"paths that name a client: {named}"

    def test_the_walk_reaches_every_kind_of_file(self) -> None:
        """Guard against the walk quietly finding nothing, or only the package."""
        found = {str(path.relative_to(PROJECT_ROOT)) for path in tracked_files()}
        for expected in ("README.md", "pyproject.toml", "src/dantron_etl/MANUAL.md", "tests/package/test_boundary.py"):
            assert expected in found, f"{expected} was not scanned"
        if (PROJECT_ROOT / ".git").exists():
            assert any(name.startswith("docs/") for name in found)
            assert any(name.startswith(".github/") for name in found)

    def test_the_local_word_list_is_never_tracked(self) -> None:
        """The list names the clients, so it must stay out of the repository."""
        assert TERMS_FILE.name not in {path.name for path in tracked_files()}

    def test_the_word_list_reads_lines_comments_and_the_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file = tmp_path / "terms"
        file.write_text("# a comment\nAcme\n\n  globex  # their ledger\nacme\n", encoding="utf8")
        monkeypatch.setenv("TERMS_UNDER_TEST", "Initech, ,umbrella")
        assert read_terms(file, "TERMS_UNDER_TEST") == ("acme", "globex", "initech", "umbrella")

    def test_no_word_list_reads_as_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TERMS_UNDER_TEST", raising=False)
        assert read_terms(tmp_path / "absent", "TERMS_UNDER_TEST") == ()


class TestThePackageNeverNamesItselfInCode:
    @pytest.mark.parametrize("source_path", _source_files(), ids=lambda p: p.name)
    def test_a_source_file_does_not_spell_the_package_name(self, source_path: Path) -> None:
        """Relative imports mean the name is never needed, so the package can be renamed in one place."""
        assert PACKAGE_NAME not in source_path.read_text(encoding="utf8").lower()


class TestVersion:
    def test_the_version_is_a_non_empty_string(self) -> None:
        import dantron_etl

        assert isinstance(dantron_etl.__version__, str) and dantron_etl.__version__.strip()


class TestTheManualDescribesTheLibraryThatExists:
    """The manual's tool table, counts, categories and function list match the library.

    Agents write pipeline files straight from the manual, so a stale name there is a step that
    fails at load.
    """

    @staticmethod
    def _manual() -> str:
        return (PACKAGE_SRC / "MANUAL.md").read_text(encoding="utf8")

    def test_the_manual_ships_inside_the_package(self) -> None:
        assert (PACKAGE_SRC / "MANUAL.md").is_file()

    def test_every_registered_tool_is_named_in_the_manual(self) -> None:
        from dantron_etl import REGISTRY

        text = self._manual()
        missing = sorted(tool.name for tool in REGISTRY.tools() if f"`{tool.name}`" not in text)
        assert not missing, f"MANUAL.md names no tool called {missing}; its tool table is stale"

    def test_the_manual_names_no_tool_that_does_not_exist(self) -> None:
        from dantron_etl import REGISTRY

        registered = {tool.name for tool in REGISTRY.tools()}
        table = self._manual().split("### 3.1 The tools that ship", 1)[-1].split("---", 1)[0]
        named = {match.strip("`") for match in re.findall(r"`[a-z_]+`", table)}
        invented = sorted(named - registered)
        assert not invented, f"MANUAL.md names {invented}, which no tool provides"

    def test_the_stated_tool_count_is_the_real_one(self) -> None:
        from dantron_etl import REGISTRY

        count = len(list(REGISTRY.tools()))
        assert f"{count} tools" in self._manual(), f"MANUAL.md does not say '{count} tools'; the registry holds {count}"

    def test_every_category_is_listed(self) -> None:
        from dantron_etl.authoring import CATEGORIES

        text = self._manual()
        missing = sorted(name for name in CATEGORIES if name not in text)
        assert not missing, f"MANUAL.md lists no category {missing}, which a new tool may declare"

    def test_every_expression_function_is_listed(self) -> None:
        from dantron_etl.expr import ALLOWED_FUNCTIONS

        text = self._manual()
        missing = sorted(name for name in ALLOWED_FUNCTIONS if f"`{name}`" not in text)
        assert not missing, f"MANUAL.md omits expression function(s) {missing}"
