"""CrossTab: turn a long frame wide. One row per key, one column per value of a key column.

The shape this exists for is a series. One row per key and period becomes one row per key with a
column per period. That is how a person reads a figure that moves over time.

**The new columns are sorted.** Not in the order the values happened to arrive. A reader scans
them across a page, so the order has to come from the values themselves.

**A value of ``names_from`` has to be text already.** A column name is text. Turning a number or a date
into text is a formatting choice with several right answers. Choosing one here would freeze it into
every cross tab this library produces. So a column of another type is refused, and the step that
formats runs in front. That also settles what sorted means: sorted as text.

**A cell with no row under it is null.** Not zero, and there is no field to make it zero. "No rows
for this key in this period" and "nothing was measured" are the same fact. A zero is a figure
somebody measured. Once a fabricated zero is in the frame, nothing downstream can tell it from a
real one. GroupBy already rules this way about summing a group whose values are all missing, and
the two tools have to agree.

That ruling has a cost, and the cost is worth stating. The value column names come from the data.
So no later step can name them in its own configuration, and nobody can write a per-column repair
after this one. The honest way to close that gap is one generic fill step, taking a frame and a
policy, written once. Until somebody asks for it the gap stays open rather than being papered over
here.

**A null or a blank in the ``names_from`` column stops the run.** Dropping those rows would delete their
figures from the tab, with nothing left to say a row had been there. Giving them a column of their
own is worse. Every name available is a name the data could also carry, so a literal "unknown"
would merge with the null group in silence. A blank fails the same way: ``""`` is not a heading a
reader can point at. Null and blank are also the same fact written two ways, one by each extract
that wrote the file, so one rule covers both. The tool refuses, names the column and counts the
rows. The fix is upstream, where an author decides whether to drop them or label them.

**A new column name that collides with a 'by' column is refused.** This is the hazard
:func:`~.authoring.check_output_column` exists for, one step removed. The name is in the rows rather
than in the configuration, so it can only be checked once the rows have been seen. Left alone, one
of the two columns wins and the other is gone. A name that collides with a column outside 'by'
is fine, because only the 'by' columns survive into the output.

**The aggregates are GroupBy's published set**, not a second copy of it. Two whitelists in one
catalog drift. A reader who learned the list from one tool would write a configuration the other
refuses. Borrowing the names means answering for every one of them here: each has a branch of its
own, so an aggregate added to that set cannot fall through to whatever this code ends with and fill
a finance grid with a plausible number under the wrong name.

**``join_text`` fills a cell the way it fills a group.** Same order -- the rows as they arrived --
same skipping of what is missing, and the same empty string for a cell whose values are all
missing. It takes GroupBy's default separator rather than a field here, because one aggregate fills
every cell of a tab and a per-cell separator would be a fifth field on a template whose four fields
are the point. A cell with *no* row under it stays null, which is this tool's ruling everywhere and
is a distinction GroupBy never has to make: a group with no rows does not exist.

**``values_from`` is required whichever aggregate is chosen.** That includes ``count``, which counts
rows and reads no column. GroupBy can tell the two apart because it names each output column. Here
one aggregate fills every cell, so the field is always there and ``count`` ignores it.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)
from .group_by import AGGREGATE_FUNCTIONS, DEFAULT_SEPARATOR


class CrossTab:
    """One row per distinct combination of 'by', one column per distinct value of `on`, holding an aggregate."""

    name = "cross_tab"
    description = "Turn `rows` wide: one row per combination of 'by', one column per value of a key column."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="The column names that stay as rows. One output row per distinct "
                "combination of them, in the order each combination first appeared.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column name; naming the same column twice counts it once.",
                ),
            ),
            "names_from": ConfigField(
                type="string",
                description="The column whose values become the new columns. It has to hold text "
                "already, and no row may leave it empty.",
                required=True,
                non_empty=True,
            ),
            "values_from": ConfigField(
                type="string",
                description="The column each cell is computed from. Required whichever aggregate "
                "is chosen; 'count' counts rows and ignores it.",
                required=True,
                non_empty=True,
            ),
            "function": ConfigField(
                type="string",
                description="How the rows that land in one cell become one value, drawn from the aggregates that ship.",
                required=True,
                enum=tuple(sorted(AGGREGATE_FUNCTIONS)),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "One output row per distinct combination of 'by', in the order each combination "
                "first appeared. One new column per distinct value of 'names_from', in sorted "
                "order, "
                "because a reader scans those columns across a page. Each cell holds the chosen "
                "aggregate of 'values_from' over the rows that landed in it. The 'by' columns come "
                "first and keep their types; the new columns take their type from the aggregate. "
                "Every other column disappears, so a cross tab is usually the last shaping step "
                "before a report."
            ),
            examples=(_totals_by_period_example(), _counts_by_period_example(), _joined_notes_example()),
            notes=(
                "A cell with no row under it is null, and there is no way to make it zero. Zero is "
                "a measured amount and null is the absence of one, and nothing downstream can tell "
                "a fabricated zero from a real one.",
                "The values of 'names_from' become column names, so that column has to hold text "
                "already. "
                "Format a number or a date into text in the step before this one, where the "
                "formatting is written down.",
                "'join_text' fills a cell the way it fills a group: the rows in the order they "
                "arrived, joined by GroupBy's separator, with the missing values skipped. A cell "
                "whose values are all missing joins to the empty string, and a cell with no row "
                "under it stays null, which is the one distinction this tool has to make and "
                "GroupBy never does.",
                "A row whose 'names_from' value is missing or blank stops the run. It cannot be "
                "given a column without inventing a name the data could also carry, and dropping it would "
                "take its figures out of the tab in silence. A blank is refused with the null "
                "because they are the same fact written two ways.",
            ),
            field_help={
                "by": "What stays as rows. Repeating a column counts it once.",
                "names_from": "What becomes columns. Its values are the new column names, sorted.",
                "values_from": "What fills the cells, before the aggregate is applied to it.",
                "function": "How several rows in one cell become one value.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The rules about the configuration alone: a column cannot be in 'by' and elsewhere.

        The 'by' columns are the ones that survive into the output as themselves. A column that is
        also the key would put its own values into the column names, and a column that is also the
        measure would be aggregated and carried forward at once. Each refusal names both fields,
        because an author looking at four fields that all name columns needs to be told which two
        overlap.
        """
        by_columns = set(_by_columns(config))
        for name in ("names_from", "values_from"):
            column = config[name]
            if column in by_columns:
                raise ConfigError(
                    f"config field {name!r} names column {column!r}, which is also in 'by'; the "
                    f"'by' columns come through as themselves, so a column cannot be a 'by' "
                    f"column and the {name!r} column at once"
                )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """What only the incoming schema can answer, read off the column names and types alone.

        A frame of a million rows and a frame of none are refused identically here. That matters
        more for this tool than for most. Waiting until a value of 'names_from' had been seen
        would let a frame with no rows through, and the run would report success having produced nothing.
        """
        frame = inputs["rows"]
        require_columns(frame, _by_columns(config), "by", "rows")
        require_columns(frame, [config["names_from"]], "names_from", "rows")
        require_columns(frame, [config["values_from"]], "values_from", "rows")

        key = config["names_from"]
        key_dtype = frame.schema[key]
        if key_dtype != pl.String:
            raise ConfigError(
                f"config field 'names_from' names column {key!r}, which holds {key_dtype}, not "
                f"text; the values of that column become column names, so format it into text in the step "
                f"before this one"
            )

        measure = config["values_from"]
        measure_dtype = frame.schema[measure]
        if config["function"] == "sum" and not measure_dtype.is_numeric():
            raise ConfigError(
                f"config field 'values_from' names column {measure!r}, which holds {measure_dtype}, not "
                f"a number, and 'function' asks for a sum"
            )
        if config["function"] == "join_text" and measure_dtype != pl.String:
            raise ConfigError(
                f"config field 'values_from' names column {measure!r}, which holds {measure_dtype}, not "
                f"text, and 'function' asks for 'join_text'; format it into text in the step "
                f"before this one, where the formatting is written down"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        key = config["names_from"]

        _refuse_unusable_keys(frame, key)
        names = sorted(frame.get_column(key).unique().to_list())
        _refuse_collisions(names, key, by_columns)

        cells = [_cell(config, name) for name in names]
        return Ports({"rows": frame.group_by(by_columns, maintain_order=True).agg(cells)})


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The 'by' columns, a column named twice counted once, in the order it was first written."""
    return list(dict.fromkeys(config["by"]))


def _refuse_unusable_keys(frame: pl.DataFrame, key: str) -> None:
    """Refuse a run where any row has no usable value in the column that becomes the column names.

    Null and blank are refused together, by one rule. They are the same fact written two ways. One
    extract leaves an unfilled period null, and the next writes it as an empty string. Splitting
    them into two rules would make the answer depend on who produced the file.

    Whitespace-only counts as blank. A heading of three spaces is no more readable than an empty
    one, and the rest of this library already reads a whitespace-only string as blank.
    """
    unusable = pl.col(key).is_null() | (pl.col(key).str.strip_chars() == "")
    offending = frame.filter(unusable)
    if offending.height:
        raise DataError(
            f"column {key!r} has no usable value on some rows, and a value that is missing or blank "
            f"cannot become a column name; drop those rows or label them before this step",
            offending=offending,
            total=frame.height,
        )


def _refuse_collisions(names: Sequence[str], key: str, by_columns: Sequence[str]) -> None:
    """Refuse a value of the key column that would take a 'by' column's name."""
    taken = set(by_columns)
    for name in names:
        if name in taken:
            raise ConfigError(
                f"column {key!r} holds the value {name!r}, which would become an output column of "
                f"that name and collide with 'by' column {name!r}; rename the 'by' column, or "
                f"change the value, because one of the two would be lost"
            )


def _cell(config: Mapping[str, Any], name: str) -> pl.Expr:
    """One value column: the aggregate of `values_from` over this cell's rows, or null if it has none."""
    here = pl.col(config["names_from"]) == name
    measure = pl.col(config["values_from"]).filter(here)
    aggregate = config["function"]

    # This guard is the ruling, and it is the line somebody will simplify away. Do not.
    #
    # An absent cell has to come back null, and the aggregate on its own will not say so. The
    # engine counts no rows as 0 and adds no rows up to 0. Neither is a figure anybody measured,
    # and once one of them is in the frame nothing downstream can tell it from a real one.
    #
    # 'sum' asks a stricter question than the rest: not "did a row land here" but "did a value".
    # A cell holding rows whose values are all missing sums to 0 too, and that is the same lie.
    # GroupBy already rules this way on the same question, and the two tools have to agree.
    measured = measure.is_not_null().any() if aggregate == "sum" else here.any()
    return pl.when(measured).then(_aggregate(measure, aggregate)).otherwise(None).alias(name)


def _aggregate(measure: pl.Expr, aggregate: str) -> pl.Expr:
    """The chosen aggregate over one cell's values, before :func:`_cell` guards it.

    `measure` is already filtered to the rows of this cell, so a row count is its length.
    """
    if aggregate == "count":
        return measure.len().cast(pl.Int64)
    if aggregate == "sum":
        return measure.sum()
    if aggregate == "count_values":
        return measure.count().cast(pl.Int64)
    if aggregate == "n_unique":
        return measure.n_unique().cast(pl.Int64)
    if aggregate == "min":
        return measure.min()
    if aggregate == "first":
        return measure.first()
    if aggregate == "join_text":
        # GroupBy's separator, not one of this tool's own. One aggregate fills every cell here, so
        # a separator would be a fifth field on a template whose four fields are the point, and a
        # grid of joined text read across a page wants the one separator a reader already knows.
        return measure.str.join(DEFAULT_SEPARATOR)
    if aggregate == "max":
        return measure.max()
    # The module docstring says an aggregate added to the published set cannot fall through to
    # whatever this function ends with; this is the line that enforces it rather than asserting it.
    # Ending on ``max`` is how ``first`` filled cells with maxima before it had a branch, and the
    # next aggregate borrowed from GroupBy would have done the same thing silently. The name is
    # already whitelisted by then, so reaching here is a library change, not an author's mistake.
    raise AssertionError(f"unreachable aggregate {aggregate!r}")  # pragma: no cover


def _totals_by_period_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "B"],
                    "period": ["p2", "p1", "p1"],
                    "amount": [5.0, 10.0, 7.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [10.0, 7.0],
                    "p2": [5.0, None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a total per code and period, with one cell holding nothing",
        configuration={"by": ["code"], "names_from": "period", "values_from": "amount", "function": "sum"},
        explanation="'period' first appears as p2 and then p1, and the columns still come back "
        "sorted, p1 before p2. Code B has no row in p2, so that cell is null. It is not zero: "
        "nobody measured anything there.",
        inputs=inputs,
        outputs=outputs,
    )


def _counts_by_period_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "A", "B"],
                    "period": ["p1", "p1", "p2", "p2"],
                    "amount": [5.0, 10.0, 7.0, 3.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": [2, None],
                    "p2": [1, 1],
                },
                schema={"code": pl.String, "p1": pl.Int64, "p2": pl.Int64},
            )
        }
    )
    return WorkedExample(
        scenario="how many rows landed in each cell, rather than what they add up to",
        configuration={"by": ["code"], "names_from": "period", "values_from": "amount", "function": "count"},
        explanation="Two rows share code A in p1, so that cell holds 2. Code B has no row in p1 at "
        "all, so its cell is null rather than 0. There was nothing there to count. 'count' reads no "
        "column, but 'values_from' is still written down, because every cell here uses one aggregate.",
        inputs=inputs,
        outputs=outputs,
    )


def _joined_notes_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "A", "B"],
                    "period": ["p1", "p2", "p1"],
                    "note": ["late", None, "checked"],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "p1": ["late", "checked"],
                    "p2": ["", None],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a grid of notes, showing the two empty cells that do not mean the same thing",
        configuration={"by": ["code"], "names_from": "period", "values_from": "note", "function": "join_text"},
        explanation="Both empty-looking cells of the p2 column are the point. Code A has a row in "
        "p2 whose note is missing, so there was something there and it said nothing: the missing "
        "value is skipped rather than joined in as a gap, nothing is left to join, and the cell is "
        "the empty string. Code B has no row in p2 at all, so the cell is null -- this tool's "
        "ruling everywhere, and the only reason the two differ, because joining the values of an "
        "empty cell would otherwise give the empty string as well.",
        inputs=inputs,
        outputs=outputs,
    )


REGISTRY.register(CrossTab())
