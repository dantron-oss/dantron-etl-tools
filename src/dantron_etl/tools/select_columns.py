"""SelectColumns: one input, one output, and only the column set (and its order) between them.

A flow that carries every column it ever read is a flow whose every later step is read against a
schema nobody can hold in their head, and narrowing that back down -- to what a later step actually
needs, or to everything except a column that has served its purpose -- is common enough to earn its
own step rather than being folded into whichever tool happens to run next. Both directions share one
config: a list of patterns and which way to read it.

**Order follows the list, under `keep`.** A pattern list is also a reordering: `[amount, code]`
puts `amount` first even though the frame does not. Under `drop` there is no list of survivors to
follow an order from, so they keep whatever order the frame already had.

**A pattern that matches nothing is refused, wildcard included.** A wildcard that silently
contributed zero columns would be a whitelist failing open -- the schema has moved under a config
written against an older one, and the author needs to be told which pattern stopped matching, not
handed back fewer columns than they asked for.

**Matching is case sensitive**, by way of :func:`fnmatch.fnmatchcase` rather than
:func:`fnmatch.fnmatch`: the latter folds case on some platforms, which would make this tool keep a
column the author did not ask for, and only on one operating system.
"""

import fnmatch
from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
)

_ACTION_ENUM: Final[tuple[str, ...]] = ("keep", "drop")


class SelectColumns:
    """Keep or drop the columns of `rows` a pattern list names; the rows themselves are untouched."""

    name = "select_columns"
    description = "Keep or drop the columns of `rows` a pattern list names, in the order asked for."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="Column-name patterns (`*` allowed, case sensitive); at least one is "
                "required and each must match at least one column.",
                required=True,
                non_empty=True,
                items=ConfigField(type="string", description="One literal column name or `*` wildcard pattern."),
            ),
            "action": ConfigField(
                type="string",
                description="Whether the patterns name the columns to 'keep' (default) or the columns to 'drop'.",
                default="keep",
                enum=_ACTION_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every pattern is matched against the frame's own columns before a single row is "
                "touched. Under 'keep', the output follows the order the patterns are listed in, not "
                "the frame's order; under 'drop', the survivors keep the frame's own order, because "
                "nothing was listed for them to follow."
            ),
            examples=(_wildcard_example(), _drop_example()),
            notes=(
                "A pattern that matches nothing is refused, wildcard included: a wildcard silently "
                "contributing zero columns would be a whitelist failing open.",
            ),
            field_help={
                "columns": "Patterns are matched case-sensitively against the frame's columns; a "
                "column matched by more than one pattern is kept only once, at its first appearance.",
                "action": "'keep' selects only the listed columns; 'drop' selects everything else, in "
                "the frame's own order.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Resolve the patterns against the incoming schema, which is where every refusal lives.

        A pattern matching nothing and a selection that empties the frame are both decided by the
        column names alone, so nothing here waits for a row. Resolving twice -- once to refuse, once
        to select -- is the price of the tool holding no state between the two calls, and it is a
        pass over a list of column names.
        """
        _selection(inputs["rows"].columns, config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        return Ports({"rows": frame.select(_selection(frame.columns, config))})


def _selection(available: Sequence[str], config: Mapping[str, Any]) -> list[str]:
    """The columns the output carries, in order -- and every refusal a pattern list can earn.

    A pure function of the column names and the configuration, so the refusals reach an author
    before any row moves and the same answer serves both the check and the work. Nothing is kept on
    the tool between the two: one registered instance serves every flow that names it.
    """
    selected: list[str] = []
    seen: set[str] = set()
    for pattern in config["columns"]:
        matches = [name for name in available if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            raise ConfigError(
                f"config field 'columns' pattern {pattern!r} matches no column of the input; the "
                f"columns present are: {', '.join(available)}"
            )
        for name in matches:
            if name not in seen:
                seen.add(name)
                selected.append(name)

    if config["action"] == "drop":
        dropped = set(selected)
        selected = [name for name in available if name not in dropped]

    if not selected:
        raise ConfigError("config field 'columns' selects nothing: 'action: drop' removed every column of the input")

    return selected


def _wildcard_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "region": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="keep a code column and its suffix ahead of the amount",
        configuration={"columns": ["code*", "amount"]},
        explanation="`code*` matches `code` and `code_suffix`, in the frame's own order; `amount` "
        "follows because it is listed second. `region` is not listed, so it does not survive.",
        inputs=inputs,
        outputs=outputs,
    )


def _drop_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "region": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A1", "B2"],
                    "code_suffix": ["x", None],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="drop the region column, keeping the rest in the frame's own order",
        configuration={"columns": ["region"], "action": "drop"},
        explanation="Only `region` matches, so it is the only column removed; the survivors keep the "
        "order the frame already had, since 'drop' names what leaves, not what stays.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(SelectColumns())
