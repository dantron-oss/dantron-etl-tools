"""``run_tool``: every call to a tool goes through here.

It checks the input ports, checks and fills the settings, runs the tool, and then checks what came
back: the output ports, the inputs left unchanged, and any per-row status. The runner, the
conformance check and the tests all run tools through ``run_tool``.

Why: docs/design/core.md#the-boundary
"""

import copy
from collections.abc import Mapping
from typing import Any

from .config import apply_defaults, validate_against_template
from .contract import Tool, call_tool_code, input_ports_for
from .errors import PortError
from .ports import Ports
from .status import STATUS_LEVELS, STATUS_SCHEMA

__all__ = ["run_tool"]


def run_tool(tool: Tool, inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Run ``tool`` on ``inputs`` with the settings ``config`` and return its output :class:`Ports`.

    In order, it:

    1. asks which input ports the step reads (from ``config`` as written) and refuses a port
       missing or extra;
    2. checks ``config`` against the tool's template, fills defaults, and calls
       ``validate_config`` on the filled settings;
    3. hands the tool a copy of each input frame and calls ``validate_inputs``, then ``execute``;
    4. refuses a result that is not :class:`Ports`, an output port missing or extra, an output
       with no columns, a change to an input's columns or row count, and a bad per-row status.

    Refusals are :class:`PortError` or :class:`ConfigError`. Any other exception from the tool's
    own code becomes an :class:`ExecutionError` naming the phase; a :class:`ToolError` the tool
    raises passes through unchanged. The outputs come back in the tool's declared order, with the
    tool's notes and status.

    Why: docs/design/core.md#the-boundary
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

    # A deep copy, so a tool that changes its settings cannot change the step or its fingerprint.
    config = copy.deepcopy(dict(config))
    template = call_tool_code(tool, "config_template", tool.config_template)
    validate_against_template(template, config)
    filled_config = apply_defaults(template, config)
    call_tool_code(tool, "validate_config", lambda: tool.validate_config(filled_config))

    # A polars clone shares the data and costs almost nothing.
    # Why: docs/design/core.md#tools-get-a-copy-of-their-inputs
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
        handed = given[name]
        if handed.schema != original.schema:
            raise PortError(
                f"{tool.name!r} changed the columns of its input port {name!r} in place; a tool must "
                f"treat its inputs as read-only (columns before: {', '.join(original.schema)}; "
                f"after: {', '.join(handed.schema)})"
            )
        if handed.height != original.height:
            raise PortError(
                f"{tool.name!r} changed the number of rows of its input port {name!r} in place, from "
                f"{original.height:,} to {handed.height:,}; a tool must treat its inputs as read-only"
            )

    _check_status(tool, result, declared_outputs)

    # Rebuilt in declared order. The notes and status must be carried across.
    return Ports(
        {name: result[name] for name in declared_outputs},
        notes=result.notes,
        status=result.status,
    )


def _check_status(tool: Tool, result: Ports, declared_outputs: tuple[str, ...]) -> None:
    """Refuse a per-row status that is on an unknown port, has the wrong shape, uses an unknown
    level, has a different height from its port, or says every row is ``ok``.

    Why: docs/design/core.md#what-the-boundary-refuses-in-a-status
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
