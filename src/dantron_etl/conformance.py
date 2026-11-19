"""The conformance check: is a tool fit to be in the catalog?

``check_tool`` holds one tool to the catalog's rules: a machine name, a description, a published
category, a well-formed template, names that survive YAML, and worked examples. Then it runs every
worked example and compares the result with what the example says it produces. ``check_registry``
does the same for a whole catalog.

**A registry** is a catalog resolved by name, and a catalog is only as trustworthy as its examples
are true. So this module ships the check that keeps it that way: every worked example a tool
documents is actually run, and its output is compared against what the tool claims it produces. An
example that has quietly drifted from its tool is worse than no example, because someone will read
it and copy it.
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


# The bare words a definition file written in YAML resolves into another type before this module
# ever sees them, in every spelling such a loader recognises. Published rather than private,
# because the person who needs the list is the tool author choosing what to call an option.
YAML_UNSAFE_VALUES: Final[frozenset[str]] = frozenset(
    {
        spelling
        for word in ("true", "false", "yes", "no", "on", "off", "y", "n", "null")
        for spelling in (word, word.capitalize(), word.upper())
    }
    | {"~", ""}
)

# Text such a loader hands over as a number. A pattern rather than a call to ``float``, so that a
# word like ``nan`` or ``inf`` -- which a loader leaves as text, and either of which is a reasonable
# name for a fill policy -- is not refused for a reason its author could do nothing about.
_NUMERIC_TEXT: Final[re.Pattern[str]] = re.compile(r"^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$")


def check_tool(tool: Tool) -> None:
    """Check ``tool`` against every rule that makes the catalog trustworthy.

    The structural rules -- a machine name, a real description, an output port unless the tool is a
    sink, unique port names, a well-formed template, documentation that matches the template -- run
    first, because there is no point running a worked example against a tool that does not even
    have a name worth reporting.

    The rule that runs last is the one that matters most: every worked example is run through
    :func:`run_tool` and its result is compared, port by port over every port the tool *declares*,
    against what the tool claims. A tool whose example has quietly drifted from its own behaviour
    is worse than one with no example at all, because someone will read the example and copy it.

    Row order is unchecked by default: a tool that pairs or groups rows produces an order that is an
    artefact of the query engine underneath, and pinning it in the catalog would make such an
    example flaky against nothing the tool actually got wrong. An example opts into the comparison
    with :attr:`WorkedExample.check_row_order`, and doing so covers every port that example
    documents, so an order-preserving tool can have the one property a reader most wants to trust
    actually checked here rather than merely claimed in prose.

    ``help()`` itself is held to one further rule: describing a tool must not change the process
    doing the describing. A snapshot is taken immediately before the call and compared immediately
    after, and a ``help()`` that exported an environment variable or changed the working directory
    while building its worked examples is refused right here, before any example runs -- a
    description that disturbs the process is the defect, not anything an example does later. The
    same comparison runs again around each example's own declared setup, in
    :func:`_check_example_with_setup`, which is this rule's mirror rather than a separate one.
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

    # A sink may be terminal, and nothing else may. A transform, a combine or a read that produces
    # nothing has been wired into a flow that can never use it, which is a defect; a sink that
    # produces nothing is the whole point of a sink -- a workbook writer has nothing honest to hand
    # on, and inventing a receipt to hand back would be shape without information. So the rule is
    # about the category rather than about every tool.
    if not tool.output_ports and tool.category != "output":
        raise ConformanceError(
            f"tool {tool.name!r} declares no output ports, and only a tool in the 'output' category "
            f"may be terminal; it is in {tool.category!r}"
        )

    # A tool says what it reads in one of two ways: a static attribute, or the optional member that
    # works the ports out from a step's configuration. Doing both makes the tool answer differently
    # depending on which one a reader asked, and the cost is silent. Configure such a tool to derive
    # no ports and the run guard accepts a supplied frame, while the decision to skip the step reads
    # the static attribute and says the step is computed. The step then runs and the supplied frame
    # is dropped, with nothing said. This used to be a line in a docstring, and a rule that holds
    # only while an author remembers it is a convention.
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

    # A tool that derives its ports can name one port twice, and nothing further down would say so.
    # Ports are built from a mapping, so a repeated name arrives as one frame: the declared ports
    # and the supplied ports collapse to the same set and the boundary sees nothing wrong. The
    # example then runs, stacks the same frame twice, and what a reader is told is that the example
    # has drifted -- which sends them to compare output rows when the defect is the port list. So
    # this check is explicit, and it runs before any example does.
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
    """One level of a template, and then whatever each of its fields describes below itself."""
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
    """One field spec, held to the same rules wherever it sits -- top level or nested.

    Nested included, because a nested declaration is where a template is least read and most
    trusted. Every rule below refuses a declaration that reads as if it constrains something and
    does not, which is the worst outcome available to a template: an element type on a field that is
    not an array, members on a field with none, ``non_empty`` on a field with no notion of
    emptiness, and the two that depend on where the field sits.

    Those two are ``default`` anywhere below the top level and ``required`` on an element spec
    itself. :func:`apply_defaults` fills the top level only -- an element that does not exist has
    nothing to fill, and inventing one would add an entry to an ordered list nobody wrote -- so a
    nested default is never applied, while reading exactly as if it were, leaving ``execute`` to
    write ``.get`` against a field its own template promised would always be there. ``required`` on
    an element spec is read by nothing at all, because an element that is absent is not an element;
    what an author means by it is ``non_empty`` on the array itself, so the refusal says so.
    ``required`` on a *member* of an element is a different declaration and stays: the validator
    enforces it per element, and an ordered-rules tool depends on it.

    A declared default is also checked against its own field here, and this is the only place it can
    be checked, because validation runs on what a caller supplied and the default is added
    afterwards. Left out, a default is the one value in a configuration that never meets the
    validator, and a default of the wrong type reaches ``execute`` as something its own template
    says it will never receive.
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
    """Whether a definition file hands this text over as something other than the text itself."""
    return text in YAML_UNSAFE_VALUES or _NUMERIC_TEXT.match(text) is not None


def _refuse_rewritten_name(tool_name: str, path: str, name: str) -> None:
    """Refuse a config field name a definition file would turn into something else.

    A field name is a key in the file. ``cross_tab`` used to have a field called ``on``, so the
    obvious way to write that step handed the loader the key ``True``. Quoting the key in the file
    works, but that puts the fix on every author of every pipeline using the tool, and nothing
    tells them. The tool has one author, so the refusal belongs here.

    Nested names included. A member of an object and a member of an element of an array are both
    keys in the file, and a template is least read where it is deepest.
    """
    if _yaml_rewrites(name):
        raise ConformanceError(
            f"tool {tool_name!r} declares a config field {path!r}, which YAML resolves to "
            f"something other than that text before this library sees it -- so a step written the "
            f"obvious way hands over a key that is not a name at all. Choose a name that survives "
            f"a definition file."
        )


def _refuse_rewritten_input_port(tool_name: str, port: str) -> None:
    """Refuse an input port name a definition file would turn into something else.

    A step says what it reads as ``inputs: {rows: earlier_step.rows}``. The port name is the key,
    on the left of the colon, which is the same place a config field name sits. A port called
    ``on`` hands the loader the key ``True``, and the name has stopped being a name.

    Quoting the key in the file works. It also puts the fix on every author of every pipeline using
    the tool, and nothing tells them. The tool has one author, so the refusal belongs here.

    Output port names are left alone. A step names one as a value, on the right of the colon, where
    a definition file hands the word over as written -- ``filter`` ships ports called ``true`` and
    ``false``.

    This covers the ports a tool declares. A tool that works its input ports out from a step's
    configuration can still end up with an unsafe name, and nothing checks that yet.
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
    """Refuse an option a definition file would hand over as something other than the word written.

    A pipeline definition is authored in YAML, which resolves a handful of bare words into other
    types before this library sees any of them, and the two ways that lands are not equally bad. An
    option arriving as ``True`` is refused by its own field's type check, which is merely confusing.
    An option arriving as ``None`` -- which is what the obvious spellings of "nothing" do -- is
    treated as a field nobody configured, filled with the field's default, and the run proceeds
    under a policy its author did not choose and was never told about. So the value is refused here,
    at authoring time, where there is a person to tell. Every such word has a safe synonym:
    ``exclude``/``include`` rather than ``false``/``true``, ``first``/``last`` rather than
    ``no``/``yes``.

    Field names are held to the same rule, by :func:`_refuse_rewritten_name`, and both read the one
    published set. Names used to be exempt on the argument that a bad one announces itself. It does
    not. A key of another type used to crash the code that reports an unknown field, and one level
    down it produced a refusal naming a field nobody had written. Declared input port names are
    held to it too, by :func:`_refuse_rewritten_input_port`, because a step names one on the left
    of a colon as well. A port worked out from a step's configuration is not checked yet. Output
    port names are exempt: a step names one as a value, where YAML leaves it alone.

    The numeric clause refuses a tool that wanted members ``"1"`` and ``"2"``, and that is correct
    rather than over-strict: such a definition arrives carrying integers, so the enum would never
    match. The fix is a field type that is not a string -- not a looser rule here.
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
    """One worked example: readable, valid, documenting every declared output, and true.

    Validating the configuration before running it is what separates two defects a single "does
    not run" refusal would report identically: an example whose configuration is wrong, and an
    example that uses a port its tool does not declare. They have two different fixes, and a
    reader told only that the example failed has to work out which one they are looking at.

    The configuration is validated against the example's own inputs too, in that same block and for
    that same reason. An example naming a column its own frames do not carry is a configuration
    nobody could ever have run, and a documented example is the thing somebody copies: reported as
    a run failure it reads as if the tool were broken, and reported as a configuration failure it
    points at the line to change. This is also why a declared setup, when there is one, has to be
    open for validation and not merely for the run that follows: resolving a connection is exactly
    the kind of thing a source's ``validate_config`` does, and a bad example must still be reported
    as a configuration defect rather than as a run failure that only happened to occur first.

    Every declared output port is compared, not merely the ports the example documents, because the
    undocumented ones are the side outputs -- the rows that did not match, the leftovers from each
    side -- which are the whole reason the envelope exists and, being anti-joins, the likeliest to
    be wrong. An example documenting one of three outputs has been run rather than checked.

    The comparison is unordered by default, and only by default: row order is an artefact of the
    query engine for a tool that pairs or groups rows, so pinning it would make such an example fail
    whenever the engine's own internals changed. ``example.check_row_order`` is passed straight
    through to the comparison, so an example that opts in still has values, dtypes, column order and
    row count checked exactly as before -- only the one additional property changes.
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
    """Check one worked example, entering its declared world first if it has one.

    ``example.setup`` is ``None`` for every one of the shipped, total-function tools, and for those
    this is exactly :func:`_run_example` and nothing more -- the fence that keeps this seam from
    changing behaviour nobody asked it to change.

    Where a setup is declared, it is entered before :func:`_run_example` is called and left open for
    the whole of it -- validation and the run both -- because validation is where a source resolves
    its connection, and closing the world after only the run would leave ``validate_config`` and
    ``validate_inputs`` looking at a process with nothing prepared. The teardown runs in a
    ``finally``, so a comparison that fails or a tool that raises still tears down the world exactly
    as a clean check would: a conformance failure that left a database or a variable behind would
    poison every check that runs after it in the same process.

    A :class:`ToolError` -- including the :class:`ConformanceError` that :func:`_run_example` itself
    raises -- is never touched here and reaches the caller exactly as raised. Only a failure of the
    setup's own entry or exit, which is never a :class:`ToolError`, is translated, because that
    failure belongs to the declaration and not to the example it was checking.

    After the world is closed, the process is compared against the snapshot taken before it was
    opened, the mirror of the same check :func:`check_tool` runs around ``help()``: a setup that put
    back everything it changed passes silently, and a teardown that left a variable set or the
    working directory moved is refused by name.
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
    """Give ``example`` a directory to work in when its configuration asks for one, then check it.

    ``example.configuration`` is the page a reader copies, so it is inspected for the ``{scratch}``
    token exactly as written, before anything else runs. A configuration that holds no token at all
    -- most of the shipped tools, and the whole of :class:`_StubTool`'s default shape in this
    module's own tests -- is delegated to :func:`_check_example_with_setup` unchanged: this is the
    fence that keeps this seam from changing behaviour nobody asked it to change.

    Where the token is present, a fresh directory is created before anything else -- before the
    example's own configuration is resolved, and before its declared ``setup``, if it has one, is
    ever entered -- because ``setup`` reaches the same directory through :func:`scratch_path`, and a
    source resolves what it is about to read as early as ``validate_config``. The configuration
    :func:`_check_example_with_setup` goes on to check is a copy with the token resolved, built with
    :func:`dataclasses.replace` rather than written into ``example`` itself, so the object this
    module was handed -- and the page it came from -- still reads ``{scratch}`` afterwards.

    The directory is removed in a ``finally``, on every path out of the check, and that removal must
    not swallow the check's own refusal: whether the directory still exists is captured before it is
    removed, and the "destroyed the checker's directory" refusal is raised only *after* the whole
    try/finally has run its course -- which means only when nothing else already raised, so a drifted
    example or a tool that threw still reports as exactly that, not as a directory gone missing.

    A fresh directory per call, never a cached one: the conformance tests in this catalog run
    :func:`check_tool` twice on purpose, and a shared directory would let a file a sink's example
    wrote on the first pass satisfy the second pass's own "nothing exists yet" without writing
    anything at all.
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
    """Check every tool in ``registry`` against :func:`check_tool`, and report every failure at once.

    Every refusal names the tool it came from, including a failure inside a tool's own template or
    validation code. One failing tool is reported as it is; several are listed in one
    :class:`ConformanceError`, so a catalog is fixed in one pass rather than one tool at a time.
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
