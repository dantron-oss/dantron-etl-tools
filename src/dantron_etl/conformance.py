"""The conformance check: is a tool fit to be in the catalog?

``check_tool(tool)`` holds one tool to the catalog's rules: a machine name, a description, a
published category, a well-formed template, names that survive YAML, and worked examples. Then it
runs every worked example and compares the result with what the example says. ``check_registry``
does the same for every tool in a registry.

Why: docs/design/core.md#worked-examples-are-run
"""

import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Final

from polars.testing import assert_frame_equal

from .boundary import run_tool
from .config import _SIZED_TYPES, FIELD_TYPES, ConfigField, _check_value, _joined, apply_defaults, validate_against_template
from .contract import CATEGORIES, Tool, WorkedExample, call_tool_code, input_ports_for, ports_from_config
from .errors import ConfigError, ConformanceError, ToolError
from .example_world import _current_scratch_directory, _process_state, _refuse_disturbance, _resolved_scratch, _uses_scratch
from .registry import ToolRegistry

__all__ = ["YAML_UNSAFE_VALUES", "check_registry", "check_tool"]


_MACHINE_NAME: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]*$")


# Words YAML reads as a boolean or null instead of text. A field name, input port name or option
# may not be one of them.
YAML_UNSAFE_VALUES: Final[frozenset[str]] = frozenset(
    {
        spelling
        for word in ("true", "false", "yes", "no", "on", "off", "y", "n", "null")
        for spelling in (word, word.capitalize(), word.upper())
    }
    | {"~", ""}
)

# Text YAML reads as a number. ``nan`` and ``inf`` stay text in YAML, so they are allowed.
_NUMERIC_TEXT: Final[re.Pattern[str]] = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def check_tool(tool: Tool) -> None:
    """Check one tool against the catalog's rules. Returns nothing; raises :class:`ConformanceError`.

    The rules, in order:

    - it has every member of :class:`Tool`, a machine name (``lower_case``), a description and a
      published category;
    - it has an output port, unless its category is ``output``;
    - it uses ``input_ports`` or ``input_ports_for``, not both, and no port name twice;
    - no field name, input port name or option is a word YAML rewrites (:data:`YAML_UNSAFE_VALUES`);
    - its template is well formed and every default passes its own field;
    - ``help()`` has a detailed description, at least one worked example, ``field_help`` only for
      real fields, and does not change environment variables or the working directory;
    - every worked example validates, documents every output port, runs, and gives exactly the
      documented output on every port.

    Why: docs/design/core.md#what-check_tool-refuses
    """
    if not isinstance(tool, Tool):
        raise ConformanceError(f"{tool!r} does not conform to the Tool protocol")

    if not _MACHINE_NAME.match(tool.name):
        raise ConformanceError(
            f"tool name {tool.name!r} is not a machine name (expected lowercase letters, digits and "
            f"underscores, starting with a letter)"
        )

    if not tool.description.strip():
        raise ConformanceError(f"tool {tool.name!r} has a blank description")

    if tool.category not in CATEGORIES:
        raise ConformanceError(
            f"tool {tool.name!r} has category {tool.category!r}, which is not one of the published "
            f"categories: {', '.join(sorted(CATEGORIES))}"
        )

    # Only an output tool may have no output ports.
    if not tool.output_ports and tool.category != "output":
        raise ConformanceError(
            f"tool {tool.name!r} declares no output ports, and only a tool in the 'output' category "
            f"may be terminal; it is in {tool.category!r}"
        )

    # Why: docs/design/core.md#input-ports-from-the-settings
    if ports_from_config(tool) and tuple(tool.input_ports):
        raise ConformanceError(
            f"tool {tool.name!r} declares static input ports ({', '.join(tool.input_ports)}) and "
            f"implements input_ports_for as well. A tool does one or the other. Either leave "
            f"input_ports empty, or drop input_ports_for and keep the static ports."
        )

    if len(set(tool.input_ports)) != len(tool.input_ports):
        raise ConformanceError(f"tool {tool.name!r} declares duplicate input port names: {tool.input_ports}")
    if len(set(tool.output_ports)) != len(tool.output_ports):
        raise ConformanceError(f"tool {tool.name!r} declares duplicate output port names: {tool.output_ports}")

    for port in tool.input_ports:
        _refuse_rewritten_input_port(tool.name, port)

    try:
        template = call_tool_code(tool, "config_template", tool.config_template)
    except ToolError as error:
        raise ConformanceError(f"tool {tool.name!r} could not build its configuration template: {error}") from error
    _check_specs(tool.name, template, prefix="")

    before_help = _process_state()
    help_ = call_tool_code(tool, "help", tool.help)
    _refuse_disturbance(f"tool {tool.name!r} disturbed the process while documenting itself", before_help)
    if not help_.detailed_description.strip():
        raise ConformanceError(f"tool {tool.name!r} has a blank detailed description")

    undocumented = set(help_.field_help) - set(template)
    if undocumented:
        raise ConformanceError(
            f"tool {tool.name!r} documents field_help for field(s) not in its config template: "
            f"{', '.join(sorted(undocumented))}"
        )

    if not help_.examples:
        raise ConformanceError(f"tool {tool.name!r} ships no worked examples")

    # Checked before any example runs; otherwise a repeated port shows up as a drifted example.
    if ports_from_config(tool):
        for example in help_.examples:
            derived = input_ports_for(tool, example.configuration)
            duplicated = sorted({port for port in derived if derived.count(port) > 1})
            if duplicated:
                raise ConformanceError(
                    f"tool {tool.name!r} derives duplicate input port name(s) {', '.join(duplicated)} "
                    f"from the configuration of its worked example {example.scenario!r}; a port "
                    f"carries one frame"
                )

    for example in help_.examples:
        _check_example(tool, template, example)


def _check_specs(
    tool_name: str,
    template: Mapping[str, ConfigField],
    prefix: str,
    *,
    nested: bool = False,
) -> None:
    """Check each field of one level of a template, and the fields nested below it."""
    for name, spec in template.items():
        _refuse_rewritten_name(tool_name, _joined(prefix, name), name)
        _check_spec(tool_name, _joined(prefix, name), spec, nested=nested)


def _check_spec(
    tool_name: str,
    path: str,
    spec: ConfigField,
    *,
    nested: bool = False,
    element: bool = False,
) -> None:
    """Check one field's declaration, at the top level or nested.

    Refuses a declaration that looks as if it constrains something and does not: an unknown type,
    a blank description, ``items`` outside an array, ``fields`` outside an object, ``non_empty``
    on a type with no notion of empty, a ``default`` below the top level (never filled), and
    ``required`` on an array element (nothing reads it). Also refuses a default its own field
    would refuse.

    Why: docs/design/core.md#template-declarations-that-do-nothing
    """
    if spec.type not in FIELD_TYPES:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} has unknown type {spec.type!r}; the types a template "
            f"may declare are: {', '.join(sorted(FIELD_TYPES))}"
        )

    if not spec.description.strip():
        raise ConformanceError(f"tool {tool_name!r} field {path!r} has a blank description")

    if spec.items is not None and spec.type != "array":
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares items but has type {spec.type!r}; an element "
            f"type constrains nothing outside an array"
        )

    if spec.fields is not None and spec.type != "object":
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares fields but has type {spec.type!r}; members "
            f"constrain nothing outside an object"
        )

    if spec.non_empty and spec.type not in _SIZED_TYPES:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares non_empty but has type {spec.type!r}, which "
            f"has no notion of emptiness; non_empty may only be declared on: {', '.join(sorted(_SIZED_TYPES))}"
        )

    if nested and spec.default is not None:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares a default that is never applied: defaults "
            f"are filled at the top level of a configuration only, because an element or a member "
            f"that nobody wrote has nothing to fill. Give the value in the tool's own code instead "
            f"of declaring it here, where the next reader will trust it."
        )

    if element and spec.required:
        raise ConformanceError(
            f"tool {tool_name!r} field {path!r} declares required on an element, which nothing "
            f"reads: an element that is absent is not an element. Declare non_empty on the array "
            f"itself to refuse a list with nothing in it."
        )

    _refuse_rewritten_enum(tool_name, path, spec)

    if spec.default is not None:
        try:
            _check_value(path, spec, spec.default)
        except ConfigError as error:
            raise ConformanceError(
                f"tool {tool_name!r} field {path!r} has a default its own declaration refuses: {error}"
            ) from error

    if spec.items is not None:
        _check_spec(tool_name, f"{path}[]", spec.items, nested=True, element=True)
    if spec.fields is not None:
        _check_specs(tool_name, spec.fields, prefix=path, nested=True)


def _yaml_rewrites(text: str) -> bool:
    """Return whether YAML would read ``text`` as something other than that text."""
    return text in YAML_UNSAFE_VALUES or _NUMERIC_TEXT.match(text) is not None


def _refuse_rewritten_name(tool_name: str, path: str, name: str) -> None:
    """Refuse a field name YAML would read as something else, at any depth.

    Why: docs/design/core.md#names-yaml-would-rewrite
    """
    if _yaml_rewrites(name):
        raise ConformanceError(
            f"tool {tool_name!r} declares a config field {path!r}, which YAML resolves to "
            f"something other than that text before this library sees it -- so a step written the "
            f"obvious way hands over a key that is not a name at all. Choose a name that survives "
            f"a definition file."
        )


def _refuse_rewritten_input_port(tool_name: str, port: str) -> None:
    """Refuse a declared input port name YAML would read as something else.

    Output port names are allowed, because a step writes them as values, which YAML leaves alone.
    Ports worked out from a step's settings are not checked.
    """
    if _yaml_rewrites(port):
        raise ConformanceError(
            f"tool {tool_name!r} declares an input port {port!r}, which YAML resolves to something "
            f"other than that text before this library sees it -- so a step wiring that port hands "
            f"over a key that is not a name at all. Choose a port name that survives a definition "
            f"file. An output port may keep such a name, because a step names one as a value, "
            f"where YAML leaves it alone."
        )


def _refuse_rewritten_enum(tool_name: str, path: str, spec: ConfigField) -> None:
    """Refuse an option (``enum`` value) YAML would read as something else.

    Use a safe word instead: ``include``/``exclude`` for ``true``/``false``. Numeric text is
    refused too; a field that wants numbers should not be a string.

    Why: docs/design/core.md#names-yaml-would-rewrite
    """
    for value in spec.enum or ():
        if isinstance(value, str) and _yaml_rewrites(value):
            raise ConformanceError(
                f"tool {tool_name!r} field {path!r} offers enum value {value!r}, which YAML resolves "
                f"to something other than that text before this library sees it -- so the value can "
                f"never arrive, and a spelling of nothing would silently become the field's default. "
                f"Choose a word that survives a definition file."
            )


def _run_example(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """Check one worked example: text present, settings valid, every output port documented, and
    the output exactly as documented.

    The settings are validated against the example's own inputs before the run, so a bad example
    is reported as a settings problem, not as a failed run. Every declared output port is compared,
    including the side outputs the example might not mention.

    Why: docs/design/core.md#every-output-port-is-compared
    """
    if not example.scenario.strip():
        raise ConformanceError(
            f"tool {tool.name!r} ships a worked example with a blank scenario; every refusal below "
            f"quotes the scenario to say which example it means"
        )
    if not example.explanation.strip():
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} has a blank explanation, which is "
            f"the thing an example exists to be"
        )

    try:
        validate_against_template(template, example.configuration)
        filled = apply_defaults(template, example.configuration)
        call_tool_code(tool, "validate_config", lambda: tool.validate_config(filled))
        call_tool_code(tool, "validate_inputs", lambda: tool.validate_inputs(example.inputs, filled))
    except ToolError as error:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} does not validate: {error}"
        ) from error

    declared = tuple(tool.output_ports)
    undeclared = set(example.outputs.names) - set(declared)
    if undeclared:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} documents output port(s) the tool "
            f"does not declare: {', '.join(sorted(undeclared))}; it declares: {', '.join(declared)}"
        )
    undocumented = set(declared) - set(example.outputs.names)
    if undocumented:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} documents no output for declared "
            f"port(s): {', '.join(sorted(undocumented))}"
        )

    try:
        produced = run_tool(tool, example.inputs, example.configuration)
    except ToolError as error:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} does not run: {error}"
        ) from error

    for name in declared:
        try:
            assert_frame_equal(produced[name], example.outputs[name], check_row_order=example.check_row_order)
        except AssertionError as error:
            raise ConformanceError(
                f"tool {tool.name!r} worked example {example.scenario!r} port {name!r} has drifted from "
                f"its documented output: {error}"
            ) from error


def _check_example_with_setup(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """Check one worked example inside its ``setup``, when it has one.

    The setup stays open for validation and the run, and is torn down even when the check fails.
    A failure of the setup itself becomes a :class:`ConformanceError`. Afterwards the process must
    be as it was: no environment variable left set, no working directory moved.
    """
    if example.setup is None:
        _run_example(tool, template, example)
        return

    before = _process_state()
    try:
        with example.setup():
            _run_example(tool, template, example)
    except ToolError:
        raise
    except Exception as error:  # noqa: BLE001 - a setup failure is translated, not swallowed
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} has a setup that failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    finally:
        _refuse_disturbance(
            f"tool {tool.name!r} worked example {example.scenario!r} left the process disturbed after "
            f"its setup was torn down",
            before,
        )


def _check_example(tool: Tool, template: Mapping[str, ConfigField], example: WorkedExample) -> None:
    """Check one worked example, first opening a ``{scratch}`` directory if its settings use one.

    The directory is new for each call and removed afterwards. The example is checked on a copy
    whose settings have the token replaced; the example itself still reads ``{scratch}``. An
    example that deletes the directory is refused, unless the check already failed for another
    reason.

    Why: docs/design/core.md#the-scratch-directory
    """
    if not _uses_scratch(example.configuration):
        _check_example_with_setup(tool, template, example)
        return

    directory = Path(tempfile.mkdtemp(prefix="etl_tool_scratch_"))
    token = _current_scratch_directory.set(directory)
    destroyed = False
    try:
        resolved_example = replace(example, configuration=_resolved_scratch(example.configuration, directory))
        try:
            _check_example_with_setup(tool, template, resolved_example)
        finally:
            destroyed = not directory.exists()
            shutil.rmtree(directory, ignore_errors=True)
    finally:
        _current_scratch_directory.reset(token)

    if destroyed:
        raise ConformanceError(
            f"tool {tool.name!r} worked example {example.scenario!r} destroyed the directory "
            f"({directory}) the checker opened for it to work in; a file left inside the directory "
            "is the example's own to leave behind, but the directory itself belongs to the checker"
        )


def check_registry(registry: ToolRegistry) -> None:
    """Run :func:`check_tool` on every tool in ``registry`` and report every failure at once.

    One failing tool raises its own error. Several are listed together in one
    :class:`ConformanceError`, so a catalog can be fixed in one pass.
    """
    failures: list[ToolError] = []
    for tool in registry.tools():
        try:
            check_tool(tool)
        except ToolError as failure:
            failures.append(failure)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        listed = "\n".join(f"  - {failure}" for failure in failures)
        raise ConformanceError(f"{len(failures)} tools do not conform:\n{listed}")
