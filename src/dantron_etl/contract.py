"""The shape every tool has: the ``Tool`` protocol and the documentation it carries.

A tool is a class with a name, a description, a category, its input and output ports, a
configuration template, help with worked examples, and three methods the boundary calls. This
module states that shape, the published categories, and how to ask a tool which ports a step reads.
It is the first file to read before writing a tool.

**A tool** is the unit that reads ports and writes ports: a machine name, a one-line description, a
category drawn from a published set, detailed help carrying worked examples, a typed configuration
template, a way to validate whatever a template alone cannot express, and the work itself.
"""

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Final, Protocol, runtime_checkable

from .config import ConfigField
from .errors import ExecutionError, ToolError
from .ports import Ports

__all__ = [
    "CATEGORIES",
    "TERMINAL_CATEGORIES",
    "DerivesInputPorts",
    "Tool",
    "ToolHelp",
    "WorkedExample",
    "catalog_input_ports",
    "call_tool_code",
    "input_ports_for",
    "ports_from_config",
]


# The groups a rendered catalog is organised into. Published and checked for the same reason as the
# types: assembled tool by tool from whatever each author typed, one group in reality becomes
# "transform", "Transform", "transformation" and "" in the rendering.
CATEGORIES: Final[frozenset[str]] = frozenset(
    {"input", "transform", "combine", "preparation", "text", "output", "assert"}
)


# The categories whose effect is not one of their ports, so a step nothing consumes is not a dead
# end when it carries one of these -- an 'output' writes somewhere else, an 'assert' raises or not.
# This is the tool's own published category, not a per-step flag an author could set and forget.
TERMINAL_CATEGORIES: Final[frozenset[str]] = frozenset({"output", "assert"})


# What a reader is told about which piece of a tool's own code broke. Named here, once, so the
# call sites that reach into tool-authored code cannot each spell the phase differently and so
# that a reader can search a log for one of these words and land on the right part of the
# boundary.
_PHASE_VERBS: Final[dict[str, str]] = {
    "config_template": "building its configuration template",
    "validate_config": "validating its configuration",
    "validate_inputs": "checking its configuration against the frames it was given",
    "execute": "running",
    "help": "documenting itself",
    "input_ports_for": "working out which input ports it reads",
}


@dataclass(frozen=True, slots=True)
class WorkedExample:
    """One demonstration of a tool: a scenario, the configuration and inputs it runs on, the output
    it claims to produce, and the explanation a reader needs to trust the claim without running it.

    The claim is not taken on faith: :func:`check_tool` runs every worked example through
    :func:`run_tool` and compares the result against ``outputs``, so a documented example is a
    checked example.

    ``check_row_order`` defaults to ``False``: for a tool that pairs or groups rows, the order they
    come back in is an artefact of the query engine, and pinning it would make an ordinary example
    fail whenever the engine's own internals changed rather than when the tool did. An example that
    documents an order-preserving tool opts in, and doing so applies to every port the example
    documents, not one.

    ``setup`` is how an example that needs a world -- a connection, a file already on disk --
    declares it, rather than a tool building that world inside ``help()`` as a side effect of
    describing itself. Left as ``None``, an example is checked exactly as one of the shipped,
    total-function tools always has been: no world entered, none to tear down.

    ``setup`` is a callable returning a context manager, not a context manager itself, and that
    indirection carries the whole guarantee. The same example is checked more than once in a
    process -- once on its own, and again every time the catalog it belongs to is checked -- so
    whatever prepares the world must be able to do so more than once. A context manager built once
    and stored, such as a bare ``@contextmanager`` generator object, is good for exactly one
    ``with`` block; asked for a second time it is already exhausted. A callable has no such limit:
    every check calls it again and gets a fresh one.

    When ``configuration`` holds the ``{scratch}`` token, :func:`check_tool` opens a directory
    before this callable is entered and resolves the token onto a private copy of ``configuration``
    for the run -- ``setup`` reaches the very same directory through :func:`scratch_path`, and the
    directory is removed again once the check returns, whatever it decided. The directory is the
    checker's, not the example's: anything a setup writes inside it goes with it, but a file written
    anywhere else is exactly as invisible to every check here as it always was, and this does not
    close that hole and is not claiming to.
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
    """The detailed half of a tool's documentation -- what its one-line description leaves out."""

    detailed_description: str
    examples: Sequence[WorkedExample] = ()
    notes: Sequence[str] = ()
    field_help: Mapping[str, str] = field(default_factory=dict)


@runtime_checkable
class Tool(Protocol):
    """The shape a step must have to be registered and run: described, and callable.

    A :class:`typing.Protocol` rather than an abstract base class, because the registry only needs
    "a described, callable thing" and structural typing says exactly that without forcing a caller
    to inherit from a class of ours. ``runtime_checkable`` gives the registry a cheap gate at
    registration time, checking that every named member is present without dictating how a caller's
    tool is implemented.

    There are two validation members rather than one because they can answer different questions.
    ``validate_config`` sees the configuration alone, so it is where a cross-field rule lives.
    ``validate_inputs`` also sees the frames, so it is where "the column this configuration names
    has to be on this port" lives -- and it is on the protocol, called on every run, rather than
    called wherever it happens to exist, because the second shape is a rule that holds while an
    author remembers. A tool with nothing to add writes ``return None``; the cost of the gate being
    real is that a tool without the member is not a :class:`Tool` at all.

    ``validate_config`` must not look at the outside world: no file, no database, no environment
    variable. The loader calls it, and a flow has to load the same way whatever is on disk, so a
    viewer or a dead-end check can read it. A check on a file belongs in ``validate_inputs``, which
    runs only when the step runs.
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
    """The optional member for a tool whose input ports depend on how the step is configured.

    Almost every tool reads a fixed set of ports and names them in ``input_ports``. One shape
    cannot: a step that writes a sheet per frame reads as many frames as its configuration names.
    Such a tool leaves ``input_ports`` empty and writes this member instead. A tool without the
    member is a :class:`Tool` like any other, and behaves exactly as it always has.

    Three rules for an author writing it. Two of them are checked, and this docstring says which.

    **A tool answers the question one way or the other, never both.** Declare static
    ``input_ports`` or implement this member. A tool doing both gives two answers to "what does
    this step read", and different parts of the library ask different ones -- so a supplied frame
    can be accepted by the one that asks the configuration and then thrown away by the one that
    reads the attribute, with nothing raised. :func:`check_tool` refuses a tool that does both.

    **Never name one port twice.** A port carries one frame, so the second use of a name replaces
    the first and a frame is lost without a word. Refused in two places: :func:`check_tool` refuses
    a worked example that does it, and the loader refuses an authored step that does it.

    **Return the ports in the order the step reads them.** The configuration handed over is the one
    a person wrote, with no default filled in -- the loader checks a step's edges before it
    validates its configuration, and the boundary checks the envelope before it fills defaults in.
    So the field naming the ports must be one the author always writes. Put the port list behind a
    default and this member sees nothing. **Nothing checks this one**, because a field that is
    absent and a field that is empty look the same from here.
    """

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]: ...


def ports_from_config(tool: Tool) -> bool:
    """Say whether this tool works its input ports out from a step's configuration."""
    return isinstance(tool, DerivesInputPorts)


def input_ports_for(tool: Tool, config: Mapping[str, Any]) -> tuple[str, ...]:
    """The input ports a step reads, given its tool and the configuration it was written with.

    Every reader of a port asks this one question, so no two of them can get different answers. A
    tool without the optional member hands back the attribute it declares, whatever the
    configuration says.
    """
    if not isinstance(tool, DerivesInputPorts):
        return tuple(tool.input_ports)
    return tuple(call_tool_code(tool, "input_ports_for", lambda: tool.input_ports_for(config)))


def catalog_input_ports(tool: Tool) -> tuple[str, ...]:
    """The input ports to show in a catalog, which has no step's configuration in hand.

    A tool with fixed ports shows them. A tool that derives its ports shows the ones its first
    worked example reads. Those names cannot drift: they come from the same member the runner asks,
    and from a configuration :func:`check_tool` executes every time it runs. They are true of that
    one example only, so show :func:`ports_from_config` beside them and say where they came from.
    """
    if not ports_from_config(tool):
        return tuple(tool.input_ports)
    examples = call_tool_code(tool, "help", tool.help).examples
    if not examples:
        return ()
    return input_ports_for(tool, examples[0].configuration)


def call_tool_code[T](tool: Tool, phase: str, action: Callable[[], T]) -> T:
    """Run one piece of a tool's own code, translating a stray failure into :class:`ExecutionError`.

    Every call site that reaches into code an author wrote -- building a template, validating a
    configuration, running against real data, describing itself -- is a call site the query engine
    can raise through, with an exception type this module's caller has never heard of. The
    ``except ToolError: raise`` arm has to come first, and it has to come first here rather than
    only around ``execute``: a tool that already raises a precise :class:`ConfigError` from inside
    its own ``validate_config`` must reach the caller as itself, not buried under a phase name.
    """
    try:
        return action()
    except ToolError:
        raise
    except Exception as error:  # noqa: BLE001 - translating whatever a tool's own code raises is the point
        raise ExecutionError(
            f"{tool.name!r} failed while {_PHASE_VERBS[phase]}: {type(error).__name__}: {error}"
        ) from error
