"""Write the tool catalog: output_files/tool_catalog.html for people, tool_catalog.json for programs.

    .venv/bin/python scripts/tool_catalog_page.py
"""

from dantron_etl.paths import project_root
from dantron_etl.viewer.catalog_page import write_catalog

OUT_DIR = project_root(__file__) / "output_files"


def main() -> None:
    html_path, json_path = write_catalog(OUT_DIR)
    print(f"-> {html_path} ({html_path.stat().st_size:,} bytes)")
    print(f"-> {json_path} ({json_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
