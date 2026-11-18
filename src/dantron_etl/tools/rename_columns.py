"""RenameColumns: a column changes its name, and nothing else about it changes.

Sibling of SelectColumns, which decides which columns survive and in what order but cannot rename
one. Without this tool a rename has to be faked with a formula that computes a new column from the
old one and a selection that drops the old one -- two steps rather than one, and the computed column
is **appended**, so the faked rename silently moves the column to the end of the frame. A consumer
reading columns positionally gets a different frame from the one it asked for and is never told. So
*the renamed column stays exactly where it was* is not a nicety here; it is the whole reason the
tool exists.

**Renames are simultaneous.** The list is one mapping applied to the input schema, not a sequence of
steps, so no entry can read another's output: `code -> region` beside `region -> code` exchanges the
two names, and `code -> region` beside `region -> territory` is a chain that needs no temporary
name. Applying the pairs one at a time would either raise a duplicate-column error from inside the
query engine or, worse, finish with two columns under one name.

**A collision is refused; a name being vacated in the same step is not a collision.** Writing over a
column that is staying is data loss with nothing said about it, so it is refused by name -- and that
check must not catch the swap above, where the name being taken is being given up in the very same
step.

**A declaration that changes nothing is refused.** Renaming a column to the name it already carries,
a blank name, a whitespace-only name: each reads as an instruction and is not one. This library
already refuses a pattern matching nothing and a `non_empty` on a type with no notion of emptiness,
for the reason that applies here too -- the usual cause is a configuration edited halfway, and
saying so beats quietly agreeing.

Every refusal is a `ConfigError` and every refusal names the offender. This
tool never reads a row, so there is no value it could diagnose: each refusal is decided from the
configuration and the incoming column names alone, which is why a zero-row frame earns exactly the
same refusals a populated one does.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
)

_FROM = "from"
_TO = "to"


class RenameColumns:
    """Rename columns of `rows` in place: same rows, same order, same dtypes, new names."""

    name = "rename_columns"
    description = "Rename columns of `rows` in place, leaving every column where it already sat."
    category = "preparation"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "columns": ConfigField(
                type="array",
                description="The renames to apply, each naming the column to rename and its new name. "
                "Applied together, so two entries may exchange names.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One rename: the column as it arrives, and the name it leaves under.",
                    fields={
                        _FROM: ConfigField(
                            type="string",
                            description="The column to rename, as it is spelled on the input port.",
                            required=True,
                        ),
                        _TO: ConfigField(
                            type="string",
                            description="The name that column carries on the output port.",
                            required=True,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each column keeps its position, its dtype and every one of its values; only the name "
                "in the header changes. The renames are resolved against the incoming columns as a "
                "single mapping, so a pair of entries may exchange two names and a chain of entries "
                "may pass a name along, neither needing a temporary name to get there."
            ),
            examples=(_in_place_example(), _exchange_example()),
            notes=(
                "A new name that collides with a column which is staying is refused, because writing "
                "over that column would lose it with nothing said; a name being renamed away in the "
                "same step is free to be taken.",
                "Renaming a column to the name it already carries is refused, as are blank and "
                "whitespace-only names: each is a declaration that does nothing, and the usual cause "
                "is a configuration edited halfway.",
            ),
            field_help={
                "columns": "Order does not matter: the list is a mapping, not a sequence of steps, and "
                "the output follows the frame's own column order rather than this list's.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Everything decidable from the configuration alone, ahead of any look at the schema.

        Blank names are refused here rather than against the frame so that the refusal can cite the
        entry's position: there is no column name worth printing, and reporting a whitespace `from`
        as "no such column '   '" would send a reader hunting for a column instead of for a typo.
        """
        _renames(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """The two refusals that need the incoming column names: a missing source, and a collision."""
        _resolved(inputs["rows"].columns, config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        mapping = _resolved(frame.columns, config)
        return Ports({"rows": frame.rename(mapping)})


def _renames(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The `(old, new)` pairs, with every refusal the configuration can earn on its own.

    Pure, and run again by :meth:`RenameColumns.execute`, so one registered instance keeps no state
    between the check and the work -- a pass over a short list of names, paid twice.
    """
    pairs: list[tuple[str, str]] = []
    taken: dict[str, int] = {}
    renamed: dict[str, int] = {}

    for index, entry in enumerate(config["columns"]):
        old, new = entry[_FROM], entry[_TO]
        for value, field in ((old, _FROM), (new, _TO)):
            if not value.strip():
                raise ConfigError(
                    f"config field 'columns[{index}]' states a blank {field!r} name; a rename must "
                    f"name a column and the name it is to carry"
                )
        if old == new:
            raise ConfigError(
                f"config field 'columns' renames {old!r} to the name it already carries; a rename "
                f"that changes nothing is a configuration edited halfway"
            )
        if old in renamed:
            raise ConfigError(
                f"config field 'columns' renames {old!r} twice, at entries {renamed[old]} and "
                f"{index}; a column has one new name or none"
            )
        if new in taken:
            raise ConfigError(
                f"config field 'columns' gives the name {new!r} to two columns, at entries "
                f"{taken[new]} and {index}; each new name must be claimed once"
            )
        renamed[old] = index
        taken[new] = index
        pairs.append((old, new))

    return pairs


def _resolved(available: Sequence[str], config: Mapping[str, Any]) -> dict[str, str]:
    """The rename mapping, checked against the columns actually arriving on the port.

    A new name collides only with a column that is **staying**: one being renamed away is vacated in
    the same step, which is what makes an exchange of two names expressible as one step.
    """
    pairs = _renames(config)
    vacated = {old for old, _ in pairs}

    for old, new in pairs:
        if old not in available:
            raise ConfigError(
                f"config field 'columns' renames {old!r}, which is not a column of the input; the "
                f"columns present are: {', '.join(available)}"
            )
        if new in available and new not in vacated:
            raise ConfigError(
                f"config field 'columns' renames {old!r} to {new!r}, but {new!r} is already a column "
                f"of the input and is not itself being renamed; it would be written over and lost"
            )

    return dict(pairs)


def _in_place_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "record_id": [11, 12],
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
                    "record_id": [11, 12],
                    "territory": ["north", "south"],
                    "amount": [10.0, -5.0],
                }
            )
        }
    )
    return WorkedExample(
        scenario="rename a column that sits in the middle of the frame",
        configuration={"columns": [{_FROM: "region", _TO: "territory"}]},
        explanation="`region` becomes `territory` and stays the second column, between `record_id` "
        "and `amount`. A formula-and-drop rename would have moved it to the end instead.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _exchange_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["X1", "Y2"],
                    "region": ["north", "south"],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "region": ["X1", "Y2"],
                    "code": ["north", "south"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="exchange two column names in one step",
        configuration={
            "columns": [
                {_FROM: "code", _TO: "region"},
                {_FROM: "region", _TO: "code"},
            ]
        },
        explanation="Both renames are applied to the incoming schema at once, so each name is free "
        "to be taken by the other column; neither entry sees the other's result, and no temporary "
        "name is needed. The values do not move -- only the headers above them.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(RenameColumns())
