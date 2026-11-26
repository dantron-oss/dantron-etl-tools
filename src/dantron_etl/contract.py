"""The ``Tool`` protocol: the shape every tool has, and the help it carries.

A tool is a class with a name, a one-line description, a category, its input and output ports, a
configuration template, help with worked examples, and three methods the boundary calls:
``validate_config``, ``validate_inputs`` and ``execute``. Read this file before writing a tool.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, TypeVar, runtime_checkable

from .config import ConfigField, expression_values
from .errors import ExecutionError, ToolError
from .expr import references
from .ports import Ports

__all__ = [
    "CATEGORIES",
    "TERMINAL_CATEGORIES",
    "DerivesColumnReferences",
    "DerivesInputPorts",
    "Tool",
    "ToolHelp",
    "WorkedExample",
    "catalog_input_ports",
    "column_references_for",
    "call_tool_code",
    "input_ports_for",
    "ports_from_config",
]


# The groups the catalog is organised into. A tool's category must be one of these.
# Why: docs/design/core.md#a-fixed-set-of-categories
CATEGORIES: Final[frozenset[str]] = frozenset(
    {"input", "transform", "combine", "preparation", "text", "output", "assert"}
)


# Steps of these categories may end a flow: an output writes somewhere else, an assert raises or not.
TERMINAL_CATEGORIES: Final[frozenset[str]] = frozenset({"output", "assert"})


# How an error message names the part of a tool's own code that failed.
_PHASE_VERBS: Final[dict[str, str]] = {
    "config_template": "building its configuration template",
    "validate_config": "validating its configuration",
    "validate_inputs": "checking its configuration against the frames it was given",
    "execute": "running",
    "help": "documenting itself",
    "input_ports_for": "working out which input ports it reads",
    "column_references": "working out which columns its expressions read",
}


@dataclass(frozen=True, slots=True)
class WorkedExample:
    """One example of a tool at work: a scenario, the settings and input frames, the output it
    gives, and a plain explanation.

    :func:`check_tool` runs every worked example and compares the result with ``outputs``, so an
    example that stops being true fails the build.

    - ``check_row_order``: compare rows in order. Leave it ``False`` for a tool that pairs or
      groups rows, whose row order is up to the query engine. It applies to every port.
    - ``setup``: a function that returns a context manager, for an example that needs a database
      or a file on disk. It is called afresh each time the example is checked.
    - A ``{scratch}`` token in ``configuration`` is replaced with a temporary directory that
      ``setup`` can reach through :func:`scratch_path`. The directory is removed after the check.

    Why: docs/design/core.md#worked-examples-are-run
    """

    scenario: str
    configuration: Mapping[str, Any]
    explanation: str
    inputs: Ports
    outputs: Ports
    check_row_order: bool = False
    setup: Callable[[], AbstractContextManager[object]] | None = None


@dataclass(frozen=True, slots=True)
class ToolHelp:
    """A tool's longer help: a detailed description, worked examples, notes and help per field."""

    detailed_description: str
    examples: Sequence[WorkedExample] = ()
    notes: Sequence[str] = ()
    field_help: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """What a class must have to be registered and run as a tool.

    Any class with these members is a tool; there is nothing to inherit from. The registry checks
    the members are present when the tool is registered.

    - ``validate_config(config)``: cross-field rules on the settings alone. It must not read a
      file, a database or an environment variable, because the loader calls it.
    - ``validate_inputs(inputs, config)``: rules that need the frames, such as "this column must
      exist". Write ``return None`` when there is nothing to check.
    - ``execute(inputs, config)``: the work. Returns a :class:`Ports` with every output port.

    Why: docs/design/core.md#the-tool-protocol
    """

    name: str
    description: str
    category: str
    input_ports: Sequence[str]
    output_ports: Sequence[str]

    def config_template(self) -> Mapping[str, ConfigField]: ...

    def help(self) -> ToolHelp: ...

    def validate_config(self, config: Mapping[str, Any]) -> None: ...

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None: ...

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports: ...


@runtime_checkable
class DerivesInputPorts(Protocol):
    """An optional member for a tool whose input ports depend on its settings.

    A tool that writes one sheet per input frame reads as many ports as its settings name. Such a
    tool leaves ``input_ports`` empty and implements ``input_ports_for(config)``, which returns the
    port names in the order the step reads them.

    Three rules:

    - Use ``input_ports`` or this member, never both. :func:`check_tool` refuses both.
    - Never return the same port name twice. :func:`check_tool` and the loader refuse it.
    - ``config`` is the settings as written, with no defaults filled in, so the field that names
      the ports must have no default. Nothing checks this one.

    Why: docs/design/core.md#input-ports-from-the-settings
    """

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]: ...


@runtime_checkable
class DerivesColumnReferences(Protocol):
    """An optional member for a tool whose brackets are not all input column names.

    Before a step runs, the runner checks that every column its expressions read is on its input,
    so a misspelling is refused by name. For most tools that is every name in brackets. A tool that
    reads its brackets another way, as multi_row_formula reads ``[Row-1:amount]`` as the column
    ``amount``, implements ``column_references(config)``. It returns, for each expression field, the
    input columns that field reads.

    Two rules:

    - Return every expression field, and every input column each one reads. A column left out is
      not checked before the step runs.
    - ``config`` is the settings as written, with no defaults filled in, as for
      :class:`DerivesInputPorts`.

    Why: docs/design/core.md#columns-read-from-the-settings
    """

    def column_references(self, config: Mapping[str, Any]) -> Mapping[str, frozenset[str]]: ...


def column_references_for(tool: Tool, config: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    """Return the input columns each expression field of a step reads: ``{field path: column names}``.

    A field with no expression is left out. Every name in brackets, unless the tool works its own
    out (:class:`DerivesColumnReferences`).
    """
    if isinstance(tool, DerivesColumnReferences):
        return dict(call_tool_code(tool, "column_references", lambda: tool.column_references(config)))
    return {path: references(text) for path, text in expression_values(tool.config_template(), config)}


def ports_from_config(tool: Tool) -> bool:
    """Say whether this tool works its input ports out from a step's configuration."""
    return isinstance(tool, DerivesInputPorts)


def input_ports_for(tool: Tool, config: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the input ports a step reads, given its tool and its settings as written.

    A tool with fixed ports returns ``input_ports`` whatever the settings say. Every part of the
    library that reads ports asks this function, so they all get the same answer.
    """
    if not isinstance(tool, DerivesInputPorts):
        return tuple(tool.input_ports)
    return tuple(call_tool_code(tool, "input_ports_for", lambda: tool.input_ports_for(config)))


def catalog_input_ports(tool: Tool) -> tuple[str, ...]:
    """Return the input ports to show in the catalog, where there is no step's settings.

    A tool with fixed ports shows them. A tool that works its ports out from its settings shows the
    ports of its first worked example, which are true of that example only. Show
    :func:`ports_from_config` beside them so the reader knows.
    """
    if not ports_from_config(tool):
        return tuple(tool.input_ports)
    examples = call_tool_code(tool, "help", tool.help).examples
    if not examples:
        return ()
    return input_ports_for(tool, examples[0].configuration)


_T = TypeVar("_T")


def call_tool_code(tool: Tool, phase: str, action: Callable[[], _T]) -> _T:
    """Run one piece of a tool's own code and return what it returns.

    ``phase`` names the piece (``"execute"``, ``"validate_config"`` ...). A :class:`ToolError` the
    tool raises passes through unchanged. Any other exception becomes an :class:`ExecutionError`
    that names the tool and the phase.
    """
    try:
        return action()
    except ToolError:
        raise
    except Exception as error:  # noqa: BLE001 - translating whatever a tool's own code raises is the point
        raise ExecutionError(
            f"{tool.name!r} failed while {_PHASE_VERBS[phase]}: {type(error).__name__}: {error}"
        ) from error
