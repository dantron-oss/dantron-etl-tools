"""Tools built on the contract in :mod:`.tool`.

Each tool's module registers its own instance into :data:`.tool.REGISTRY` at import time, so
importing a tool's module -- directly, or by importing this package -- is what puts it in the
shipped catalog. This package imports every module beneath it itself, discovered from the
filesystem rather than listed here, so that importing this package (or the top-level package, which
imports this one) always sees the whole catalog, not merely whatever tool a caller happened to
import by name.
"""

import importlib
import pkgutil

for _module_info in pkgutil.iter_modules(__path__):
    if not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")
