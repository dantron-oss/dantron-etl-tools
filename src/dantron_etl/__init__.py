"""A generic, client-agnostic data-transformation tool library.

This package provides small, composable transformation tools (filter, join, group-by, and similar
building blocks) that any pipeline can assemble into a larger data-transformation flow, regardless
of which system produced the data or which system consumes it. It is destined to be extracted into
its own standalone distribution, installable and reusable outside of the repository it started in.

The one rule that governs everything in this package: it must never depend on code belonging to
the application that happens to house it today, because this package is meant to ship, unmodified,
into other, unrelated repositories.
"""

from . import tools

__version__ = "0.1.0.dev0"

__all__ = ["__version__", "tools"]
