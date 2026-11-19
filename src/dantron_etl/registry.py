"""The catalog of tools, looked up by name.

``REGISTRY`` is the shipped catalog. Each module under ``tools/`` adds its tool to it when it is
imported, and importing the package imports them all. ``get_tool(name)`` is the short way to look
one up.
"""

from collections.abc import Iterator
from typing import Final

from .contract import Tool
from .errors import RegistryError

__all__ = ["REGISTRY", "ToolRegistry", "get_tool"]


class ToolRegistry:
    """A catalog of tools looked up by name.

    ``register(tool)`` adds one and refuses a second tool with the same name, or an object that
    is not a :class:`Tool`. ``get(name)`` returns one, or raises :class:`RegistryError` listing
    what there is. Iterating gives the names in sorted order.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise RegistryError(f"{tool!r} does not conform to the Tool protocol and cannot be registered")
        if tool.name in self._tools:
            raise RegistryError(f"a tool named {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            available = ", ".join(self.names()) if self._tools else "none"
            raise RegistryError(f"no tool named {name!r} is registered; registered tools are: {available}") from None

    def names(self) -> tuple[str, ...]:
        """Return the registered names, sorted, so the catalog keeps a stable order."""
        return tuple(sorted(self._tools))

    def tools(self) -> tuple[Tool, ...]:
        """Return the registered tools, in :meth:`names` order."""
        return tuple(self._tools[name] for name in self.names())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())


# The shipped catalog. Each module under ``tools/`` registers its tool here when imported.
REGISTRY: Final[ToolRegistry] = ToolRegistry()


def get_tool(name: str) -> Tool:
    """Return the shipped tool called ``name``. Short for ``REGISTRY.get(name)``."""
    return REGISTRY.get(name)
