"""The source package holds the library, its tests and its example, and no local file besides.

Hatchling builds a source package from the working tree, so anything lying in the checkout -- a
`.env`, a key, a client's CSV under `pipelines/` -- went into it. The build is run here, offline,
on a copy of this checkout with such files planted in it. Skipped where `uv` or a cached build
backend is not available.
"""

import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PLANTED = (
    ".env",
    ".env.local",
    ".env.example",
    "id_rsa",
    "server.pem",
    "pipelines/client/orders.csv",
    "pipelines/client.yaml",
    "src/dantron_etl/.env",
    "tests/.env.test",
    "output_files/run.parquet",
    "notes.txt",
    ".forbidden-terms",
)


@pytest.fixture(scope="module")
def sdist_names(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed, so the source package cannot be built here")

    checkout = tmp_path_factory.mktemp("checkout") / "repo"
    shutil.copytree(
        ROOT,
        checkout,
        ignore=shutil.ignore_patterns(".git", ".venv", "output_files", "__pycache__", "*_cache", "dist"),
    )
    for name in PLANTED:
        planted = checkout / name
        planted.parent.mkdir(parents=True, exist_ok=True)
        planted.write_text("do-not-ship\n", encoding="utf-8")

    built = subprocess.run(
        [uv, "build", "--sdist", "--offline", "--out-dir", str(checkout / "dist")],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if built.returncode != 0:
        pytest.skip(f"the source package could not be built offline here: {built.stderr.strip()[-300:]}")

    (archive,) = (checkout / "dist").glob("*.tar.gz")
    with tarfile.open(archive) as opened:
        return {member.name.split("/", 1)[1] for member in opened.getmembers() if "/" in member.name}


@pytest.mark.parametrize("name", PLANTED)
def test_a_local_file_lying_in_the_checkout_is_left_out(sdist_names: set[str], name: str) -> None:
    assert name not in sdist_names


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "README.md",
        "src/dantron_etl/__init__.py",
        "src/dantron_etl/MANUAL.md",
        "tests/package/test_sdist.py",
        "tests/fixtures/workbook_with_pivot.xlsx",
        "src/dantron_etl/viewer/templates/walkthrough.html",
        "src/dantron_etl/viewer/scaffold/pipelines/example.yaml",
        "pipelines/fixture.yaml",
        "pipelines/fixture/settings.yaml",
        "pipelines/fixture/entries.csv",
    ],
)
def test_the_library_its_tests_and_its_example_are_in(sdist_names: set[str], name: str) -> None:
    assert name in sdist_names


def _ignored_by_git(name: str) -> bool:
    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout, so .gitignore cannot be asked")
    asked = subprocess.run([git, "check-ignore", "--no-index", "-q", name], cwd=ROOT, timeout=30)
    return asked.returncode == 0


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "src/dantron_etl/.env", "id_rsa", "server.pem", "run.parquet", ".forbidden-terms"]
)
def test_git_ignores_a_local_secret_or_data_file(name: str) -> None:
    assert _ignored_by_git(name)


@pytest.mark.parametrize("name", [".env.example", "pipelines/fixture.yaml", "tests/fixtures/workbook_with_pivot.xlsx"])
def test_git_still_sees_a_template_and_the_example_flow(name: str) -> None:
    assert not _ignored_by_git(name)
