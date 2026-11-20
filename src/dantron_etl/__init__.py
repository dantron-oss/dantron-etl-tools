"""A library of generic data transformation tools, wired together by a YAML pipeline file.

Everything needed to load and run a flow is importable from here::

    flow = read_pipeline("pipelines/sales.yaml")
    trace = run_pipeline(flow)
    trace.frame("total_by_month.rows")

You also get what a run hands back, the tool catalog (``REGISTRY``, ``get_tool``), focus on one
entity, and the errors, all under ``ToolError``. A tool author imports from ``authoring`` instead.
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
