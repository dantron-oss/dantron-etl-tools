"""Serve whatever has been written under output_files/runs, and the source behind it.

Separated from the scripts that BUILD a run, because the two have nothing to do with each other and
coupling them made restarting the viewer cost a full rebuild. A run is written by whatever built it;
this only hands out what is already on disk.

    .venv/bin/python scripts/serve_runs.py
"""

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dantron_etl.paths import project_root  # noqa: E402
from dantron_etl.preview import serve  # noqa: E402

# Resolved from this script's own location rather than from the library's -- see `dantron_etl.paths`.
# Set ETL_PROJECT_ROOT to serve the runs of a checkout other than the one this file sits in.
PROJECT_ROOT = project_root(__file__)
RUNS_ROOT = PROJECT_ROOT / "output_files" / "runs"
HOST_PORT = 8899


def main() -> None:
    # The source root is the repository, so a page can show the code behind a step. Read-only, and
    # every path is resolved and checked to have landed inside it -- see `preview.serve`.
    serve(RUNS_ROOT, port=HOST_PORT, source_root=PROJECT_ROOT)

    runs = sorted(p.name for p in RUNS_ROOT.iterdir() if (p / "ports.json").is_file()) if RUNS_ROOT.is_dir() else []
    print(f"serving {len(runs)} run(s) on http://localhost:{HOST_PORT}/")
    for name in runs:
        print(f"    http://localhost:{HOST_PORT}/{name}/")
    print("\n  ctrl-c to stop")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
