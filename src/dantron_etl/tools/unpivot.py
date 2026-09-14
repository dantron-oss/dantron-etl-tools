"""Unpivot: turn a wide frame long. Every column outside the index becomes a name and a value.

CrossTab turns long into wide, and until this tool there was no way back. That asymmetry is the
whole justification for adding a tool at all: a frame that arrives one column per period cannot be
filtered by period, joined on period or grouped by it, because the period is a heading rather than
a value anywhere in the frame. One step that melts those headings into a column puts them back
where every other tool in this catalog can read them.

**There is no field naming the columns to melt.** Every column is either named in ``index``, in
which case it stays as it is and repeats on every output row, or it is melted. There is no third
category, so a list of the melted columns would say nothing ``index`` does not already say -- and
would be the one place two halves of a configuration could disagree about the same frame. The
second reason matters more: the obvious name for such a field is ``on``, and a definition file
reading ``on: [p1, p2]`` hands this library the boolean ``True`` standing where a name should be.
That trap has cost this project twice, and a field that does not exist cannot be named ``on``.

**``index`` is required and cannot be empty.** An unpivot with no identifier produces a column of
names and a column of values with nothing to say which row each came from, and nothing can be
joined back to it. A flow that genuinely wants that writes a RecordId in front, which is one step
and leaves the identifier visible.

**``into`` and ``values`` are required and have no default.** The two new columns are the whole
point of the reshape, and a default would publish a column called something nobody chose into the
frame every later step and every report reads from.

**The melted columns must agree on type, except that a column that is all nulls takes the others'
type.** This is Union's ruling, copied deliberately: an unpivot stacks columns on each other the
way Union stacks frames on each other, so it is the same question and it gets the same answer.
Text stacked under a number gives a column of text that used to be money, and whichever one got
rewritten did so in silence, so it is refused -- and the refusal names both columns and both types,
because "they differ" without saying how sends a reader back to the data to find out which. A
column of nothing is the exception because null is the one value belonging to every family, and an
unpopulated column is an ordinary frame rather than a mistake.

**A flow melts a disagreeing set on purpose by saying what the values are.** ``values_type`` names
the type every melted column is read as, in the spelling ``cast_columns`` uses, and while it is
given the refusal above no longer applies. The shape it exists for is a block that gains a column
every period with unrelated text beside it: the text shifts position as the block grows, so nothing
in the configuration can name it in ``index``, and telling it apart from a period means reading what
its heading says -- which needs the melt to have happened first. Stating the type is what keeps the
figures figures. Melted as text they come back as whatever a renderer shortened them to, and a
figure a hair under a half-cent comes back a hair over it and rounds the other way.

**Converting a column to the type it already holds is accepted here, and produces no note.**
``cast_columns`` refuses exactly that, and the difference is what each tool is given. There the
columns are the ones a flow named one by one, so one already holding the stated type is a line
somebody forgot to delete. Here the melt set is everything the flow did *not* name -- and the whole
reason for stating a type is that the flow cannot enumerate that set -- so "some of them are already
right" is the ordinary case rather than a configuration edited halfway, and refusing it would refuse
the feature.

**A value that will not convert stops the run, unless the flow says otherwise**, in the words
``cast_columns`` settled. ``on_invalid: error`` is what happens when the field is absent, and it
names the column, the offending values and the type asked for; ``on_invalid: empty`` leaves them
null and reports how many, out of how many, and what share, which raises the step to a warning. The
count is taken *before* the values are replaced, because afterwards a null that failed to convert
and a null that was always there are the same null. One note per melted column that actually lost
something: a clean column is not news, and here most of them are clean. ``on_invalid`` without
``values_type`` is refused -- nothing converts, so whichever word was written would do nothing at
all, and a field that silently does nothing is worse than a missing one.

**Row order is column-major**: all the rows of the first melted column, then all the rows of the
second, in the frame's own column order. Not interleaved, which is what a naive reshape gives, and
not sorted, which would order the output by something the author never wrote down. The melted
columns are read in the order the frame carries them for the same reason CrossTab sorts the columns
it creates: each tool takes the one order that is a fact about what it was given.

**An ``index`` that covers every column is refused.** Nothing is left to melt, so the step reshapes
nothing and the wide frame travels on to steps written for a long one. That is the reading
``cast_columns`` already gives a cast to the type a column already holds: a configuration edited
halfway, where saying so beats agreeing.

**A null in a melted cell survives as a row holding nothing.** The row existed and the cell was
empty, and both facts are data. Dropping it would take a pair out of the long frame with nothing
left to say it had been there, and a later count by name would be short without looking short.

**Every refusal is read off the schema**, so a frame of a million rows and a frame of none are
refused identically. An unpivot whose configuration only fails once a row has been seen is an
unpivot that reports success on the day a filter upstream matched nothing.

**The index columns come back in the order they were configured**, not the order the frame happens
to carry them in. A pipeline file is read to find out what a step produces, and the configuration
is the author's own statement of the output's shape; taking the frame's order instead would make
the output depend on something that is not written down anywhere in the file.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Final

import polars as pl

from ..column_types import DECLARED_DTYPE, DECLARED_TYPES
from ..tool import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)

# The words `cast_columns` publishes for the same question, in the same spelling and the same order.
# 'empty' rather than 'null': `null` unquoted in a definition file IS the null value, so an author
# reaching for the obvious word would hand over None and the boundary would read it as unset.
_ON_INVALID_ENUM: Final[tuple[str, ...]] = ("error", "empty")


class Unpivot:
    """Turn `rows` long: every column outside `index` becomes a row naming it and holding its cell."""

    name = "unpivot"
    description = "Turn `rows` long: every column outside `index` becomes a row naming it and holding its cell."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "index": ConfigField(
                type="array",
                description="The column names that stay as they are and repeat on every output "
                "row, in the order the output carries them. Every other column is melted, and "
                "there is deliberately no field listing those.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column name; naming the same column twice counts it once.",
                ),
            ),
            "into": ConfigField(
                type="string",
                description="The name of the new column holding the melted column's own name. "
                "Always text, and it may not take the name of a column already present.",
                required=True,
                non_empty=True,
            ),
            "values": ConfigField(
                type="string",
                description="The name of the new column holding the melted cell. It takes the "
                "melted columns' type, or the type named in 'values_type', and may not take the "
                "name of a column already present.",
                required=True,
                non_empty=True,
            ),
            "values_type": ConfigField(
                type="string",
                description="The type every melted column is read as, spelled the way the query "
                "engine spells it. Left out, the melted columns must already agree on a type; "
                "given, each is read as this one and a disagreement is no longer refused.",
                enum=DECLARED_TYPES,
            ),
            # No declared default, and 'error' is the behaviour when the field is absent rather
            # than a value filled in for it. The cross-field rule below reads the *filled*
            # configuration, so a default of 'error' would make every configuration look as though
            # its author had written the field, and 'on_invalid' without 'values_type' could never
            # be refused at all.
            "on_invalid": ConfigField(
                type="string",
                description="What a value that will not convert to 'values_type' does: 'error' "
                "stops the run (what happens when this is left out), 'empty' leaves it null and "
                "reports how many, out of how many. Refused without 'values_type' beside it.",
                enum=_ON_INVALID_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "One output row per input row per melted column. The index columns come first, in "
                "the order they were configured, keeping their types and repeating once per melted "
                "column; then 'into', holding the melted column's name as text; then 'values', "
                "holding the cell at the melted columns' own type, or at the type 'values_type' "
                "states. Rows come back column-major: all the rows of the first melted column, then "
                "all the rows of the second, in the frame's own column order. This is the inverse "
                "of 'cross_tab'."
            ),
            examples=(_periods_become_rows(), _some_columns_stay_put(), _a_stated_type_melts_a_mixed_block()),
            notes=(
                "There is no field naming the columns to melt. Every column is either in 'index' "
                "or melted, so such a list would say nothing new, could contradict 'index', and "
                "would want to be called 'on' -- a word a definition file reads as the boolean "
                "true where a column name should be.",
                "The melted columns must agree on type, except that a column that is all nulls "
                "takes the others' type. That is Union's ruling about stacking frames, applied to "
                "stacking columns, for the same reason: null is the one value belonging to every "
                "family. Columns that genuinely differ are refused, naming both columns and both "
                "types.",
                "A null in a melted cell comes back as a row holding nothing, never as no row. "
                "The row existed and the cell was empty, and both are facts about the data.",
                "Row order is column-major and is not sorted, so a step needing another order puts "
                "a Sort after it, as everywhere else here.",
                "An 'index' that covers every column is refused: the step would reshape nothing "
                "and hand a wide frame on to steps written for a long one.",
                "'values_type' states the type every melted column is read as, which is how a block "
                "whose columns disagree is melted deliberately -- the case being a block that gains "
                "a column every period, with text beside it that therefore cannot be named in "
                "'index'. Reading it as the type it is keeps a figure a figure; reading the lot as "
                "text loses a digit to whatever shortens it later.",
                "'on_invalid' says what a value that will not convert to 'values_type' does: "
                "'error' stops the run, 'empty' leaves it null and reports how many out of how "
                "many, raising the step to a warning. It is refused without 'values_type', because "
                "there is nothing for it to govern. Reading a column as the type it already holds "
                "is ordinary here and says nothing, unlike 'cast_columns': the melt set is "
                "everything the flow did not name.",
            ),
            field_help={
                "index": "What stays as it is and repeats. Everything else is melted.",
                "into": "The new column holding the melted column's name, always text.",
                "values": "The new column holding the cell, at the melted columns' type.",
                "values_type": "The type every melted column is read as; without it they must agree.",
                "on_invalid": "What a value that will not convert does. Only beside 'values_type'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The rules the configuration answers alone: the two new columns are two columns, and new.

        An index column comes through as itself, so a new column taking its name means one of the
        two is lost and nothing says which. ``into`` and ``values`` colliding with each other is the
        same fault written in one field instead of two. Both refusals name the field to change,
        because an author looking at three fields that all name columns has to be told which.

        The last rule is the cross-field one: ``on_invalid`` governs a conversion, and without
        ``values_type`` there is no conversion to govern, so whichever word the author wrote would
        do nothing at all and the run would report success. Both fields are named in the refusal,
        and so is the fix, since either one of the two could be the one that was meant.
        """
        index = set(_index_columns(config))
        for field in ("into", "values"):
            column = config[field]
            if column in index:
                raise ConfigError(
                    f"config field {field!r} names new column {column!r}, which is also in "
                    f"'index'; the index columns come through as themselves, so one of the two "
                    f"would be lost"
                )

        if config["into"] == config["values"]:
            raise ConfigError(
                f"config fields 'into' and 'values' both name column {config['into']!r}; they are "
                f"two columns -- the melted column's name and its cell -- and cannot share one name"
            )

        if config["on_invalid"] is not None and config["values_type"] is None:
            raise ConfigError(
                f"config field 'on_invalid' is given as {config['on_invalid']!r} with no "
                f"'values_type' beside it; it says what becomes of a value that will not convert, "
                f"and with no stated type nothing converts -- state a 'values_type', or drop "
                f"'on_invalid'"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Everything the incoming schema settles, decided before a row moves.

        Which columns are melted is read off the column names, so a frame with no rows is refused
        for exactly the reasons a populated one is -- and, where it is not refused, hands back the
        whole schema rather than whatever the rows happened to reveal.
        """
        frame = inputs["rows"]
        index = _index_columns(config)
        require_columns(frame, index, "index", "rows")

        melted = _melted_columns(frame, index)
        if not melted:
            raise ConfigError(
                f"config field 'index' names every column on input port 'rows', so nothing is left "
                f"to melt and the step would reshape nothing; the columns present are "
                f"{', '.join(frame.columns)}"
            )

        # Only the melted names can collide here. An `into` or `values` colliding with an index
        # column was already refused above, and a melted column's name is gone from the output, so
        # this is the collision that would quietly take a name the frame is still using.
        for field in ("into", "values"):
            column = config[field]
            if column in melted:
                raise ConfigError(
                    f"config field {field!r} names new column {column!r}, which input port 'rows' "
                    f"already carries and is about to melt; write to a name nothing uses"
                )

        # Called for its refusal, not its answer: the types have to be settled off the schema, or a
        # frame whose rows all happened to be filtered away would be accepted and one whose rows
        # survived would not. A stated 'values_type' is the flow taking that decision itself, so
        # there is nothing left to refuse -- each column is read as the type it names.
        if config["values_type"] is None:
            _settled_dtype(frame, melted)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        index = _index_columns(config)
        melted = _melted_columns(frame, index)

        if config["values_type"] is None:
            # Cast rather than take each column as it is: an all-null column takes the others' type
            # here, so the stacked result carries the type the data actually has instead of losing
            # it to the one column that had none.
            settled = _settled_dtype(frame, melted)
            values = {column: frame[column].cast(settled) for column in melted}
            notes: list[str] = []
        else:
            values, notes = _stated_values(frame, melted, config["values_type"], config["on_invalid"] == "empty")

        # One frame per melted column, stacked in the frame's own column order. That is the
        # column-major ruling expressed directly rather than produced and then re-ordered: there is
        # no intermediate order for a later change to leak through. The index columns are selected
        # in the order they were configured, which is the order the output carries them in.
        parts = [
            frame.select(
                *(pl.col(name) for name in index),
                pl.lit(column, dtype=pl.String).alias(config["into"]),
            ).with_columns(values[column].alias(config["values"]))
            for column in melted
        ]
        return Ports({"rows": pl.concat(parts, how="vertical")}, notes=notes)


def _index_columns(config: Mapping[str, Any]) -> list[str]:
    """The index, a column named twice counted once, in the order it was written."""
    return list(dict.fromkeys(config["index"]))


def _melted_columns(frame: pl.DataFrame, index: Sequence[str]) -> list[str]:
    """Everything the index does not keep, in the frame's own column order.

    Read off ``frame.columns`` rather than off the rows, which is what makes a frame with no rows
    hand back the whole schema instead of the index columns and nothing else.
    """
    kept = set(index)
    return [name for name in frame.columns if name not in kept]


def _settled_dtype(frame: pl.DataFrame, melted: Sequence[str]) -> pl.DataType:
    """The one type the melted columns share, or a refusal naming the two that do not.

    Union's ruling, and the comparison is written the same way: a column whose type is ``Null`` has
    no type to disagree with, so it is passed over here and adopts whatever the others settle on.
    Where every melted column is like that there is nothing to adopt, and ``Null`` is the honest
    answer rather than a refusal -- that is the shape a run has on the day everything upstream
    matched nothing, which is the day it most needs to keep its shape rather than stop.
    """
    typed = [(name, frame.schema[name]) for name in melted if frame.schema[name] != pl.Null]
    if not typed:
        return pl.Null

    first_name, settled = typed[0]
    for name, dtype in typed[1:]:
        if dtype != settled:
            raise ConfigError(
                f"the columns about to be melted give different types, so one of them would be "
                f"rewritten into the other on the way through: {first_name} is {settled} and "
                f"{name} is {dtype}; cast one of them in the step before this one, or name it in "
                f"'index' so it is not melted at all"
            )
    return settled


def _stated_values(
    frame: pl.DataFrame,
    melted: Sequence[str],
    declared: str,
    lenient: bool,
) -> tuple[dict[str, pl.Series], list[str]]:
    """Each melted column read as the stated type, and a note for each one that lost something.

    ``cast_columns``' answer to the same question, copied whole and down to the wording of the note:
    two tools answering one question similarly rather than identically is how a reader ends up
    trusting the rule they learned second. What differs is which columns are worth reporting. A
    column that converts cleanly says nothing here, because most of them do -- the melt set is
    everything the flow did not name, so a note apiece would bury the one that lost a value.
    """
    target = DECLARED_DTYPE[declared]
    values: dict[str, pl.Series] = {}
    notes: list[str] = []

    for column in melted:
        if lenient:
            # Counted before the values are replaced, because afterwards the evidence is gone: a
            # null that failed to convert and a null that was always there look identical, and the
            # whole point of this mode is to say how many of the first there were.
            #
            # Unguarded, like `cast_columns`' lenient path, and with the same known hole: a pair the
            # engine refuses for the whole column rather than value by value -- a Date read as a
            # Float64 -- arrives from here as an ExecutionError rather than a DataError. Closing it
            # belongs to a change that closes it in `cast_columns` too, or the two tools answer one
            # question differently again.
            read = frame[column].cast(target, strict=False)
            lost = int((frame[column].is_not_null() & read.is_null()).sum())
            if lost:
                share = lost / frame.height * 100 if frame.height else 0.0
                notes.append(
                    f"{lost:,} of {frame.height:,} values in {column!r} ({share:.2g}%) did not "
                    f"read as {declared} and are now empty"
                )
            values[column] = read
            continue
        try:
            values[column] = frame[column].cast(target)
        except Exception as error:
            # A refusal about data rather than about configuration: stating the type is a legal
            # thing to ask for, and what stopped the run is the values. That is the difference
            # between "go and look at the flow" and "go and look at the rows". The column is the
            # only thing that says where to look, since no field of this tool names a melted one.
            #
            # The examples are gathered with a non-strict cast, which answers null for the values
            # that would not convert -- and is itself wrapped, because a pair the engine refuses
            # outright would otherwise throw here, turning a refusal into a crash inside the
            # reporting of it.
            try:
                offending = frame.filter(
                    pl.col(column).is_not_null() & pl.col(column).cast(target, strict=False).is_null()
                )
            except Exception:  # noqa: BLE001 - the report is best-effort; the refusal is not
                offending = frame.head(0)
            raise DataError(
                f"melted column {column!r} does not read as {declared}: {error}",
                offending=offending,
                total=frame.height,
            ) from error
    return values, notes


def _periods_become_rows() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {"code": ["A", "B"], "p1": [10.0, 7.0], "p2": [5.0, None]},
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B"],
                    "period": ["p1", "p1", "p2", "p2"],
                    "amount": [10.0, 7.0, 5.0, None],
                },
                schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="a column per period becomes a row per period, so a period can be filtered on",
        configuration={"index": ["code"], "into": "period", "values": "amount"},
        explanation="Both melted columns come back whole and in the frame's own order: all of p1, "
        "then all of p2. B's p2 cell was empty and arrives as a row holding nothing rather than as "
        "no row, because the row existed and the cell was blank and both are facts. Nothing names "
        "p1 and p2 in the configuration -- every column outside 'index' is melted.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _some_columns_stay_put() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "site": ["north", "south"],
                    "opening": [10.0, 7.0],
                    "closing": [12.0, 6.0],
                },
                schema={"code": pl.String, "site": pl.String, "opening": pl.Float64, "closing": pl.Float64},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B"],
                    "site": ["north", "south", "north", "south"],
                    "measure": ["opening", "opening", "closing", "closing"],
                    "amount": [10.0, 7.0, 12.0, 6.0],
                },
                schema={"code": pl.String, "site": pl.String, "measure": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="keep a descriptive column out of the reshape by naming it in the index",
        configuration={"index": ["code", "site"], "into": "measure", "values": "amount"},
        explanation="'site' is text and would be melted with the two figures -- and refused for "
        "disagreeing with them on type -- if it were not named in 'index'. Naming it there is the "
        "only way to keep a column out of the melt, which is why an index that names every column "
        "is a configuration somebody stopped editing halfway.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _a_stated_type_melts_a_mixed_block() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [10.0, 7.0],
                    "p2": [5.0, 2.0],
                    "comment": ["checked", None],
                },
                schema={"code": pl.String, "p1": pl.Float64, "p2": pl.Float64, "comment": pl.String},
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"] * 3,
                    "period": ["p1", "p1", "p2", "p2", "comment", "comment"],
                    "amount": [10.0, 7.0, 5.0, 2.0, None, None],
                },
                schema={"code": pl.String, "period": pl.String, "amount": pl.Float64},
            )
        }
    )
    return WorkedExample(
        scenario="melt a block whose columns disagree on type by stating the type they are read as",
        configuration={
            "index": ["code"],
            "into": "period",
            "values": "amount",
            "values_type": "Float64",
            "on_invalid": "empty",
        },
        explanation="'comment' is text and cannot be named in 'index', because the number of "
        "columns in front of it changes from one load to the next and nothing written down can say "
        "where it will be. Stating 'values_type' melts the lot as figures; 'on_invalid: empty' "
        "leaves the text holding nothing and reports how many values that was, out of how many, so "
        "the run carries on and the size of it carries on with it. A later step drops the rows "
        "whose name is not a period.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Unpivot())
