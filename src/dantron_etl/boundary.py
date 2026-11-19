"""``run_tool``: the one door every call to a tool goes through.

It checks the ports a tool was given, checks and fills the configuration, runs the tool, and then
checks what came back: the ports, the inputs left untouched, and any per-row status. A stray
exception from a tool's own code comes out as an ``ExecutionError`` naming the tool and the phase.
The runner, the conformance check and the tests all run tools through here.

**The boundary calls into a tool's own code five times per run**, in an order that is part of the
contract: ask which input ports this step reads, build the template, validate the configuration,
check that configuration against the schemas of the frames that actually arrived, and only then run.
The fourth is the least obvious and was the last to exist: neither a template nor a cross-field rule
has a frame in front of it, so "does the column this configuration names exist, on the port it will
be read from?" had nowhere to live but the first lines of every tool's own ``execute`` -- which is a
rule that holds while every author remembers it, and therefore not a rule.
"""

from collections.abc import Mapping
from typing import Any

from .config import apply_defaults, validate_against_template
from .contract import Tool, call_tool_code, input_ports_for
from .errors import PortError
from .ports import Ports
from .status import STATUS_LEVELS, STATUS_SCHEMA

__all__ = ["run_tool"]


def run_tool(tool: Tool, inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Run ``tool`` on ``inputs`` with ``config``, checking the envelope on the way in and out.

    The order below is part of the contract, not an implementation detail. Ports come first,
    because a port mistake belongs to the flow that wired the step rather than to the configuration
    anybody wrote. The template comes next, and it is checked here rather than left to the tool: a
    tool with nothing to add beyond its template writes an empty ``validate_config``, which is the
    natural thing to write and which used to buy it no validation whatsoever. A tool's own
    ``validate_config`` is therefore for what a template cannot express, it runs last, and it runs
    on the *filled* configuration, so a cross-field rule reads a defaulted value by subscript
    instead of guessing with ``.get``. ``execute`` never sees a configuration that fails
    validation -- a caller relying on that is relying on something this function guarantees. A tool
    that derives its ports is therefore asked about the configuration as written, before any
    default is filled in, which is the same configuration the loader asks it about.

    On the way out, what came back must be :class:`Ports`. Returning a bare mapping of frames is
    the likeliest authoring slip, and left unchecked it surfaces as an attribute error raised from
    inside this function, naming neither the tool nor the mistake.

    The tool is handed a copy of each input frame, made with polars' ``clone``, which shares the
    data and costs nothing. A change the tool makes in place -- rows appended, a value set, a column
    added or dropped -- lands on the copy, so it never reaches the caller's frame, another branch
    reading the same port, or a step's record of what it produced. A change to the copy's columns
    or its number of rows is still refused afterwards, because it is a slip a tool author should
    hear about. A value changed in place on the copy is not looked for: it cannot reach anything but
    this tool's own result, and comparing every value would cost a pass over every input per step.

    Between the tool's own configuration rules and ``execute`` sits ``validate_inputs``, the only
    phase that sees the configuration and the frames together. It runs on the filled configuration,
    like ``validate_config``, and it exists because the question it answers -- do the columns this
    configuration names exist, on the ports they will be read from -- can be answered by neither of
    its neighbours, and so used to be answered by the first few lines of every tool's ``execute``,
    in each tool's own words, after this boundary had already reported that the tool was running.

    Each call into the tool's own code -- ``input_ports_for``, ``config_template``, ``validate_config``,
    ``validate_inputs``, ``execute`` -- goes through :func:`call_tool_code`, so a stray exception
    from any of them, including one raised from inside the query engine a tool called into, arrives
    here as :class:`ExecutionError` naming which phase failed, rather than as whatever exception
    type that phase happened to raise. Each phase names itself distinctly, because a reader sent to
    the wrong one of two neighbouring functions has been told nothing. A :class:`ToolError` a tool
    raises on purpose is never touched by this -- including a :class:`DataError`, which is a
    diagnosis about the rows and would be buried by a phase name: it reaches the caller exactly as
    the tool wrote it.
    """
    declared_inputs = input_ports_for(tool, config)
    declared_input_set = set(declared_inputs)
    supplied = set(inputs.names)

    undeclared_inputs = supplied - declared_input_set
    if undeclared_inputs:
        raise PortError(
            f"{tool.name!r} was given undeclared input port(s) {', '.join(sorted(undeclared_inputs))}; "
            f"it declares: {', '.join(declared_inputs)}"
        )
    missing_inputs = declared_input_set - supplied
    if missing_inputs:
        raise PortError(
            f"{tool.name!r} is missing declared input port(s) {', '.join(sorted(missing_inputs))}; "
            f"it declares: {', '.join(declared_inputs)}"
        )

    template = call_tool_code(tool, "config_template", tool.config_template)
    validate_against_template(template, config)
    filled_config = apply_defaults(template, config)
    call_tool_code(tool, "validate_config", lambda: tool.validate_config(filled_config))

    # The tool is handed its own copy of each input frame. A polars clone shares the data and costs
    # nothing (0.01 ms on a million rows), and a change made in place to the copy -- rows appended,
    # a value set, a column added -- never reaches the caller's frame or another branch reading it.
    given = Ports({name: frame.clone() for name, frame in inputs.items()}, notes=inputs.notes, status=inputs.status)
    call_tool_code(tool, "validate_inputs", lambda: tool.validate_inputs(given, filled_config))

    result = call_tool_code(tool, "execute", lambda: tool.execute(given, filled_config))
    if not isinstance(result, Ports):
        raise PortError(
            f"{tool.name!r} handed back {type(result).__name__} rather than Ports; a tool must return "
            f"its outputs as named ports"
        )

    declared_outputs = tuple(tool.output_ports)
    declared_output_set = set(declared_outputs)
    produced = set(result.names)

    undeclared_outputs = produced - declared_output_set
    if undeclared_outputs:
        raise PortError(
            f"{tool.name!r} produced undeclared output port(s) {', '.join(sorted(undeclared_outputs))}; "
            f"it declares: {', '.join(declared_outputs)}"
        )
    missing_outputs = declared_output_set - produced
    if missing_outputs:
        raise PortError(
            f"{tool.name!r} never produced declared output port(s) {', '.join(sorted(missing_outputs))}; "
            f"it declares: {', '.join(declared_outputs)}"
        )

    for name in declared_outputs:
        if result[name].width == 0:
            raise PortError(
                f"{tool.name!r} output port {name!r} has no columns at all; an output must carry its "
                f"schema even when it carries no rows"
            )

    for name, original in inputs.items():
        copy = given[name]
        if copy.schema != original.schema:
            raise PortError(
                f"{tool.name!r} changed the columns of its input port {name!r} in place; a tool must "
                f"treat its inputs as read-only (columns before: {', '.join(original.schema)}; "
                f"after: {', '.join(copy.schema)})"
            )
        if copy.height != original.height:
            raise PortError(
                f"{tool.name!r} changed the number of rows of its input port {name!r} in place, from "
                f"{original.height:,} to {copy.height:,}; a tool must treat its inputs as read-only"
            )

    _check_status(tool, result, declared_outputs)

    # Rebuilt so the ports come back in the tool's declared order whatever order it produced them
    # in -- and carrying the notes and the per-row status across, which rebuilding would otherwise
    # drop. A tool that reported something worth knowing and had it silently discarded at the
    # boundary is the exact failure the boundary exists to prevent, one level up, and it has
    # happened here once already: the notes were dropped by this very rebuild.
    return Ports(
        {name: result[name] for name in declared_outputs},
        notes=result.notes,
        status=result.status,
    )


def _check_status(tool: Tool, result: Ports, declared_outputs: tuple[str, ...]) -> None:
    """Refuse a per-row status that cannot mean what it says, before anything downstream reads it.

    Checked on every call rather than trusted, because the correspondence between a status row and
    a data row is positional and is never re-derived afterwards: a status one row short does not
    report less, it reports the *wrong* rows, sliding every mark after the gap onto its neighbour,
    and nothing built from it later could tell. Both directions are refused, since a check written
    against the short case alone passes a status a tool built per *input* row after filtering.

    The shape is refused too, and so is a level outside :data:`STATUS_LEVELS`: what reads a status
    is a renderer, and a level it has no drawing for shows as neither a tick nor a warning nor a
    cross, so the row shows nothing and reads as fine -- the same defect as a missing column,
    arriving more quietly.

    An all-``ok`` status is refused last and on sight. Seven million rows asserting that nothing
    happened would make the presence of a status mean nothing, and a mark that is always there is a
    mark nobody reads. This is the ruling this library already takes on a cast to the type a column
    already holds: an inert declaration is refused rather than agreed with.
    """
    for name, status in result.status.items():
        if name not in declared_outputs:
            raise PortError(
                f"{tool.name!r} reported a per-row status for {name!r}, which is not one of its "
                f"output ports; it declares: {', '.join(declared_outputs)}"
            )

        if status.schema != STATUS_SCHEMA:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} in the wrong shape: "
                f"expected {dict(STATUS_SCHEMA)}, got {dict(status.schema)}"
            )

        unknown = sorted(str(level) for level in status["level"].unique().to_list() if level not in STATUS_LEVELS)
        if unknown:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} holding level(s) "
                f"{', '.join(unknown)}, which nothing can draw; the published levels are: "
                f"{', '.join(STATUS_LEVELS)}"
            )

        height = result[name].height
        if status.height != height:
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} of {status.height:,} "
                f"row(s) against a port of {height:,}; the two line up row for row, and a status of "
                f"another height marks the wrong rows rather than fewer of them"
            )

        if not (status["level"] != "ok").any():
            raise PortError(
                f"{tool.name!r} reported a per-row status for port {name!r} saying only that every "
                f"row is fine; a status says what a step degraded, and one that degraded nothing "
                f"reports no status at all"
            )
