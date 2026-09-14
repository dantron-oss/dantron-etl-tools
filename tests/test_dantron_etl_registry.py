"""Tests for the shipped tool catalog: importing the library must fill it.

Every tool module registers its own instance into ``REGISTRY`` at import time, at the bottom of the
module. That is a fine place to register from and a terrible thing to rely on, because it means the
catalog is only as complete as the set of modules something happened to import by name. A consumer
who writes the obvious line --

    from dantron_etl.tool import REGISTRY

-- imports ``dantron_etl`` and ``dantron_etl.tool`` and nothing else, and gets a catalog with zero
tools in it. Nothing raises. ``check_registry(REGISTRY)`` passes, because every rule it enforces
holds vacuously over an empty catalog, and a generated catalog page renders blank. That is the
defect this file exists to pin, and it is pinned in a subprocess: inside a pytest session the tool
modules have already been imported by the tests that exercise them, so an in-process assertion
about what an import pulls in would pass no matter what the package does -- the same trap the
import-boundary tests document and dodge the same way.

The decision this file also fixes: **importing ``dantron_etl`` alone must populate the catalog**,
not merely importing ``dantron_etl.tools``. Two reasons. The consumer line above is the one people
write, and it runs the top-level package's ``__init__`` and no tool module, so populating only from
``dantron_etl.tools`` leaves the reported defect unfixed for the import that caused it. And the fix
for that consumer -- adding an otherwise-unused ``import dantron_etl.tools`` beside their real
import -- is exactly the line a linter deletes and a reviewer waves through. The cost is that
``import dantron_etl`` becomes an eager import of every tool module; that cost is nearly nothing
here, because the contract module every tool imports already pulls in the query engine, which is
the only heavyweight dependency in the package.

What ships is discovered from the filesystem rather than listed here, so a tool added later is
covered by these tests without anybody remembering to add it -- and the convention that makes the
discovery honest, one tool per module named after the tool it holds, is itself pinned below.

One consequence of the subprocess rule is worth stating outright: the two tests that make a claim
about what an import pulls in are red when this file runs on its own *and* when the whole suite
runs, while the in-process tests below are red only on their own -- in a full run a sibling test
module has already imported the tool modules by name and filled the catalog for them. That is not
a flaw in those tests, it is the reason the population claims are not among them.
"""

import os
import subprocess
import sys
from itertools import permutations
from pathlib import Path

import pytest

import dantron_etl.tools
from dantron_etl.tool import REGISTRY, check_registry, check_tool

TOOLS_DIR = Path(dantron_etl.tools.__file__).parent

_READ_CATALOG = """
{importing}
from dantron_etl.tool import REGISTRY
print(",".join(REGISTRY.names()))
"""


def _shipped_tool_names() -> tuple[str, ...]:
    """The machine name of every tool the package ships, read off the files under ``tools/``.

    Discovered by walking the filesystem, not hardcoded, so a tool added in later work is covered
    by every test in this file automatically. The walk assumes one tool per module, named after its
    module -- the convention pinned by
    :meth:`TestShippedCatalog.test_every_shipped_module_registers_the_tool_named_after_it`, which is
    what lets a name be known without importing anything.
    """
    return tuple(sorted(path.stem for path in TOOLS_DIR.rglob("*.py") if not path.stem.startswith("_")))


def _catalog_after(importing: str) -> tuple[str, ...]:
    """The names in ``REGISTRY`` in a fresh interpreter that ran ``importing`` and then read it.

    A fresh interpreter is the whole point: this session has already imported the tool modules, so
    an in-process check would report a populated catalog regardless of what the package's own
    imports do. The reading line is deliberately the consumer's own line, and it imports nothing a
    tool registers from, so it cannot mask the result it is reporting.
    """
    env = os.environ.copy()
    search_path = os.pathsep.join(entry for entry in sys.path if entry)
    env["PYTHONPATH"] = search_path

    result = subprocess.run(
        [sys.executable, "-c", _READ_CATALOG.format(importing=importing)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"a fresh interpreter running {importing!r} failed (exit={result.returncode}):\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    printed = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    return tuple(printed.split(",")) if printed else ()


class TestShippedCatalog:
    """Importing the library fills the catalog, and what is in it is what the package ships."""

    @pytest.mark.parametrize(
        "importing",
        [
            pytest.param("", id="a_consumer_importing_only_the_registry"),
            pytest.param("import dantron_etl", id="importing_the_package"),
            pytest.param("import dantron_etl.tools", id="importing_the_tools_package"),
        ],
    )
    def test_the_catalog_is_populated_however_the_library_is_imported(self, importing: str) -> None:
        """None of these three lines may hand back an empty catalog.

        The first case is the reported defect verbatim: a consumer who asks the contract module for
        the registry and reads a blank page. The other two are the imports somebody would reach for
        after being told the first one does not work, and they have to agree with it, because a
        catalog whose contents depend on which import a caller guessed is not a catalog.
        """
        assert _catalog_after(importing) == _shipped_tool_names()

    @pytest.mark.parametrize("order", ["-".join(names) for names in permutations(("package", "tools", "module"))])
    def test_importing_the_package_and_a_tool_module_in_any_order_registers_each_tool_once(
        self,
        order: str,
    ) -> None:
        """Six import orders, six identical catalogs, and no duplicate-name refusal from any of them.

        Registering from a module and importing that module from a parent package is a cycle, and
        the two ways it goes wrong are opposites: a tool imported twice through two paths raises
        ``a tool named ... is already registered`` and takes the whole import down, or the cycle
        resolves against a half-initialised package and raises before the registration runs. A
        consumer reaching straight for a tool module is the order most likely to trip it, and it is
        also the order nothing else in the test suite exercises.
        """
        first_module = _shipped_tool_names()[0]
        statements = {
            "package": "import dantron_etl",
            "tools": "import dantron_etl.tools",
            "module": f"import dantron_etl.tools.{first_module}",
        }
        importing = "\n".join(statements[name] for name in order.split("-"))

        assert _catalog_after(importing) == _shipped_tool_names()

    @pytest.mark.parametrize("name", _shipped_tool_names())
    def test_the_tool_each_module_ships_is_named_after_it_and_conforms(self, name: str) -> None:
        """One tool per module, its machine name is its module's name, and it satisfies the catalog.

        The naming convention is worth a rule rather than a habit, because it is what makes the
        catalog answerable from the filesystem: a reader who wants to know what the library can do
        lists a directory, and this file's own discovery does exactly that. A module shipping a
        second tool, or a tool whose name and module have drifted apart, fails here and gets a
        decision made about it rather than going unnoticed.

        Parametrised by name so a failure says which tool broke, and ``check_tool`` is the half that
        matters: it runs each tool's worked examples, and an example nobody runs is worse than no
        example, because somebody will copy it.

        This one runs in-process, unlike the two above, and it is deliberately not the test that
        pins *population*: within one pytest session the catalog can have been filled by any other
        test module that imported a tool by name, so an in-process claim about what an import pulls
        in cannot be trusted. What is honest in-process is everything about the tool object itself,
        once resolving it has succeeded.
        """
        assert name in REGISTRY, f"{name}.py ships a tool that importing the library does not register"
        tool = REGISTRY.get(name)
        assert tool.name == name

        check_tool(tool)

    def test_the_whole_catalog_conforms_and_is_not_empty_while_doing_so(self) -> None:
        """``check_registry`` iterates the catalog, so on an empty one it checks nothing and passes.

        That is the second half of the reported defect: the catalog-wide conformance call that a
        consumer or a build step would rely on is green on an empty registry, and green for the
        reason that there is nothing in it. Pinning the size of the catalog against the number of
        tool modules on disk -- a number known to be non-empty -- is what turns that vacuous pass
        into a meaningful one: the call below is known to have visited every shipped tool.
        """
        shipped = _shipped_tool_names()

        assert shipped, f"no tool modules found under {TOOLS_DIR}; the discovery walk has gone blind"
        assert REGISTRY.names() == shipped
        assert len(REGISTRY) == len(shipped)

        check_registry(REGISTRY)
