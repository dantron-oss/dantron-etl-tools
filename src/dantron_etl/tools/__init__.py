"""The tools. Importing this package registers every one of them.

Each module in this folder registers its tool into `REGISTRY` when imported. This package imports
every module it finds here, except helpers whose name starts with `_`, so the catalog is always whole.
"""

import importlib
import pkgutil

for _module_info in pkgutil.iter_modules(__path__):
    if not _module_info.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module_info.name}")
