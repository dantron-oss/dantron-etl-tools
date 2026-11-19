"""Everything a tool author needs, in one import.

A tool file imports from here and nowhere else in the core: the ``Tool`` shape, ``Ports``, the
configuration and help classes, the helpers that check a configuration against a frame, the per-row
status helpers, the errors a tool raises, and ``REGISTRY`` to register itself with. The names live
in smaller modules; this one only gathers them. ``tools/filter.py`` is the smallest complete
example of a tool.
"""

from .boundary import run_tool
from .config import FIELD_TYPES, ConfigField, require_not_blank
from .conformance import YAML_UNSAFE_VALUES, check_registry, check_tool
from .contract import CATEGORIES, TERMINAL_CATEGORIES, DerivesInputPorts, Tool, ToolHelp, WorkedExample
from .errors import (
    MAX_REPORTED_ROWS,
    ConfigError,
    ConformanceError,
    DataError,
    ExecutionError,
    PortError,
    RegistryError,
    ToolError,
)
from .expr.dates import MAX_YEAR, MIN_YEAR, count_between, date_or_null, days_in_month
from .example_world import SCRATCH_TOKEN, example_database, scratch_path, temporary_environment
from .expr import references
from .frame_checks import (
    REPLACE_FIELD,
    check_output_column,
    compile_for,
    compile_predicate_for,
    compiled_dtype,
    require_columns,
)
from .ports import Ports
from .registry import REGISTRY, ToolRegistry
from .status import (
    STATUS_LEVELS,
    STATUS_REASON,
    STATUS_SCHEMA,
    STATUS_VALUE_CHARS,
    row_status,
    unreadable_report,
    value_caption,
)

__all__ = [
    # the shape of a tool
    "CATEGORIES",
    "FIELD_TYPES",
    "TERMINAL_CATEGORIES",
    "ConfigField",
    "DerivesInputPorts",
    "Ports",
    "Tool",
    "ToolHelp",
    "WorkedExample",
    # checking a configuration against the frames a tool was given
    "REPLACE_FIELD",
    "check_output_column",
    "compile_for",
    "compile_predicate_for",
    "compiled_dtype",
    "references",
    "require_columns",
    "require_not_blank",
    # counting between dates and building them, shared with the expression functions
    "MAX_YEAR",
    "MIN_YEAR",
    "count_between",
    "date_or_null",
    "days_in_month",
    # per-row status
    "STATUS_LEVELS",
    "STATUS_REASON",
    "STATUS_SCHEMA",
    "STATUS_VALUE_CHARS",
    "row_status",
    "unreadable_report",
    "value_caption",
    # the errors a tool raises
    "MAX_REPORTED_ROWS",
    "ConfigError",
    "ConformanceError",
    "DataError",
    "ExecutionError",
    "PortError",
    "RegistryError",
    "ToolError",
    # the world a worked example runs in
    "SCRATCH_TOKEN",
    "example_database",
    "scratch_path",
    "temporary_environment",
    # registering, running and checking a tool
    "REGISTRY",
    "YAML_UNSAFE_VALUES",
    "ToolRegistry",
    "check_registry",
    "check_tool",
    "run_tool",
]
