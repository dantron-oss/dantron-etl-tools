"""A generic, client-agnostic data-transformation tool library.

This package provides small, composable transformation tools (filter, join, group-by, and similar
building blocks) that any pipeline can assemble into a larger data-transformation flow, regardless
of which system produced the data or which system consumes it. It is destined to be extracted into
its own standalone distribution, installable and reusable outside of the repository it started in.

The one rule that governs everything in this package: it must never depend on code belonging to
the application that happens to house it today, because this package is meant to ship, unmodified,
into other, unrelated repositories.

Everything needed to load and run a flow is importable from here: ``read_pipeline``,
``run_pipeline``, what a run hands back, the catalog, and the errors. A tool author imports from
the ``authoring`` module instead.
"""

from . import tools
from .column_types import DECLARED_TYPES as COLUMN_TYPES
from .errors import (
    ConfigError,
    ConformanceError,
    DataError,
    ExecutionError,
    PipelineError,
    PortError,
    RegistryError,
    RunError,
    ToolError,
)
from .expr import ExpressionError
from .focus import FocusSpec, RelatedKey, slice_trace
from .loader import dead_ends, load_pipeline, read_pipeline
from .paths import project_root
from .registry import REGISTRY, get_tool
from .runner import run_pipeline
from .spec import Pipeline, PortRef, Step
from .trace import CachedFrame, ColumnCheckError, PipelineTrace, StepEvent, StepFailedError

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "tools",
    # load and run a flow
    "dead_ends",
    "load_pipeline",
    "read_pipeline",
    "run_pipeline",
    # what a flow is, and what a run hands back
    "CachedFrame",
    "Pipeline",
    "PipelineTrace",
    "PortRef",
    "Step",
    "StepEvent",
    # following one entity through a run
    "FocusSpec",
    "RelatedKey",
    "slice_trace",
    # the catalog, and where a project lives
    "COLUMN_TYPES",
    "REGISTRY",
    "get_tool",
    "project_root",
    # the errors, one family under ToolError
    "ColumnCheckError",
    "ConfigError",
    "ConformanceError",
    "DataError",
    "ExecutionError",
    "ExpressionError",
    "PipelineError",
    "PortError",
    "RegistryError",
    "RunError",
    "StepFailedError",
    "ToolError",
]
