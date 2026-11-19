"""The viewer: a finished run drawn as a page, and the local server that shows it.

Not imported by ``import`` of the library itself. A flow that only runs never loads it.

- ``build`` and ``build_panels`` draw one run as a page and its per-step panels.
- ``save_run`` saves a run a project's own script made, so ``view`` lists and shows it.
- ``project`` finds a project's pipelines; ``runs`` runs and saves them; ``server`` shows them.
- ``layout`` places the steps; ``render`` writes the HTML; ``assets`` reads the template files.
- ``catalog_page`` draws the tool catalog.
"""

from .render import build, build_panels
from .runs import save_run

__all__ = ["build", "build_panels", "save_run"]
