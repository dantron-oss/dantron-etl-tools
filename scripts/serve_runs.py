"""Serve whatever has been written under output_files/runs, and the source behind it.

Separated from the scripts that BUILD a run, because the two have nothing to do with each other and
coupling them made restarting the viewer cost a full rebuild. A run is written by whatever built it;
this only hands out what is already on disk.

    .venv/bin/python scripts/serve_runs.py
"""

import sys
import time
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
    # The source root is the repository, so a page can show the code behind a step. Only YAML,
    # Python and Markdown files inside it are served, never a hidden one such as .env -- see
    # `preview.serve`. It listens on this machine only; pass host="0.0.0.0" to serve() to change that.
    server = serve(RUNS_ROOT, port=HOST_PORT, source_root=PROJECT_ROOT)

    runs = sorted(p.name for p in RUNS_ROOT.iterdir() if (p / "ports.json").is_file()) if RUNS_ROOT.is_dir() else []
    print(f"serving {len(runs)} run(s) on http://localhost:{HOST_PORT}/")
    for name in runs:
        print(f"    http://localhost:{HOST_PORT}/{name}/")
    print("\n  ctrl-c to stop")
    # Parked on a sleeping loop rather than on Event().wait(). A wait() blocks the main thread
    # inside a C-level lock, and CPython can only run the handler that raises KeyboardInterrupt
    # when the main thread comes back up into the interpreter -- which it never does. So ctrl-c is
    # ignored and the only way out is to close the terminal, which is exactly what was reported.
    # Waking twice a second costs nothing and gives the signal somewhere to land.
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    # Said before the shutdown rather than after, because shutdown takes a moment and a reader who
    # pressed ctrl-c wants to see that it was heard.
    print("\nstopping")
    server.shutdown()
    server.server_close()


if __name__ == "__main__":
    main()
