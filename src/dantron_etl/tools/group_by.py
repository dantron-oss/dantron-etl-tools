"""GroupBy: one row per distinct key, carrying only the columns the configuration asks for.

**An aggregate is configuration; the thing aggregated is an expression.** The expression language is
row-wise, and its own whitelist deliberately excludes aggregates, so widening it to accept
``sum([amount])`` would make that spelling legal in exactly one config field and meaningless in
every other -- a second dialect a reader has to learn from where a string happens to sit. So this
tool publishes its own small whitelist, :data:`AGGREGATE_FUNCTIONS`, and ``aggregations[].function``
is an enum *derived* from it, so the two cannot drift apart and the whitelist cannot fail open.

**Group order follows first appearance**, not a sort: a multi-column key has no obvious total
order, and first appearance is the order the reader of the input already holds. polars promises
nothing about group order unless asked, so this needs ``maintain_order=True`` rather than trusting
an accident of the engine.

**A null in a ``by`` column is a group of its own**, agreeing with RemoveDuplicates and mirroring
what this tool is named after -- SQL's ``GROUP BY``, which groups nulls together. Dropping such rows
would delete their figures from every total downstream with nothing left in the output to say a row
had been there. This is the opposite of Join's ruling for the same-looking question, because a join
mirrors ``=``, under which a null pairs with nothing.

**A sum of nothing is null, not zero.** polars sums an all-null group to ``0``; passed on, that would
put a figure nobody measured into a report, and nothing downstream could tell it from a real zero.
Nulls are still skipped within a group holding any value at all, so ``[10, null]`` sums to ``10`` --
which is what makes "every value missing" the one case that produces null.

**``first`` is the only aggregate here that depends on the order rows arrived in**, and it is the
only one that can give two answers for the same data. It exists because a group sometimes carries a
column that is the same on every row of it -- a document number on the lines of one document -- and
taking any of them is taking all of them. When that is true, ``first``, ``min`` and ``max`` agree and
the choice does not matter.

When it is *not* true, ``first`` answers with whichever row happens to be first, which is a fact
about the order and not about the data. So a flow using it on a column that might vary should say so,
and a flow that needs the answer to be repeatable puts a Sort in front of this step -- exactly as
everywhere else here. A group whose supposedly-constant column is not constant is worth counting with
``n_unique`` beside it, which is how that assumption gets checked instead of assumed.

**``count`` counts rows; ``count_non_null`` counts the values that are there.** The two differ and get
two names rather than one name and a flag: ``count`` takes no ``value`` at all, because a row count
cannot depend on a column, and ``count_non_null`` requires one and ignores its nulls. The legacy
dataframe library spells the non-null count ``count``, so an author porting a step from it would
otherwise get the other number without being told.

**Every column that is neither grouped nor aggregated disappears.** ``min`` and ``max`` are the only
way to carry forward a column that is constant within a group, which is why those two ship even
though nothing else needs them for arithmetic.

**``join_text`` makes one string out of a group's several**, which is how a group's detail survives
a step whose whole job is to throw detail away. It is SQL's ``string_agg``: the values in the order
the rows arrived -- so it reads that order exactly as ``first`` does, and a flow wanting another one
puts a Sort in front -- joined by a ``separator`` that defaults to ``", "``. Missing values are
skipped, so ``["a", null, "b"]`` joins to ``"a, b"`` rather than ``"a, , b"``, which would read as a
value that is blank instead of one that is absent.

**A group with nothing to join is the empty string**, which is deliberately the opposite of the
ruling ``sum`` makes about the same group. Joining no strings together produces the empty string in
any language that has both, while adding no numbers together produces a total nobody measured.

``separator`` is refused beside any other aggregate rather than ignored -- the way Join refuses
``keys`` beside ``left_keys`` -- because an author who wrote ``min`` where they meant ``join_text``
would otherwise get a separator that vanishes and a column of single values where they expected a
list. What is joined has to be text, refused on the schema, because a number or a date joined into a
string ships as a plausible-looking string of digits that is unsortable and unsummable from there on.
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    compile_for,
    compiled_dtype,
    references,
    require_columns,
    require_not_blank,
)

# The published aggregate whitelist, mirroring how ``expr.ALLOWED_FUNCTIONS`` is published: a
# template author reads this rather than discovering the boundary by trial and error, and the
# enum below is derived from it rather than typed out beside it, so the two cannot drift apart.
# It is read outside this module too: CrossTab offers the same aggregates by importing this set
# rather than keeping a second copy, so adding or removing one here changes both tools.
# ``mean`` is absent on purpose -- no step wants it, and its denominator silently depends on how
# nulls are treated, which is exactly the kind of question this tool refuses to answer quietly.
AGGREGATE_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"sum", "count", "count_non_null", "n_unique", "min", "max", "first", "join_text"}
)

# The one aggregate that reads no column at all: a row count is a property of the group, not of any
# value in it, so it is the sole member excluded from "needs a value".
_NO_VALUE_FUNCTION: Final[str] = "count"

# The one aggregate that reads a second field: the name is here so the rules about it read as rules
# rather than as a string compared in four places.
_JOINING_FUNCTION: Final[str] = "join_text"

# The only place the separator's default is written down. A default declared inside a template
# element is never filled in -- defaults are filled at the top level of a configuration alone -- so
# writing ``", "`` there as well would be two defaults that can disagree. CrossTab reads this one
# too, for the same reason it reads the whitelist above rather than keeping a second copy: a reader
# who learned what a joined cell looks like from one tool gets the same thing from the other.
DEFAULT_SEPARATOR: Final[str] = ", "

# The whole-number types polars adds up into themselves, and the totals each can hold. polars
# lets a total past these limits wrap round to a wrong number without a word. So a sum of one of
# these is worked out as Int64, checked against the limits, and handed back in its own type: the
# column keeps the type it always had, and a total that does not fit stops the run. The smaller
# whole-number types are not here because polars already adds those up into Int64. CrossTab reads
# this too.
NARROW_SUM_LIMITS: Final[dict[pl.DataType, tuple[int, int]]] = {
    pl.Int32: (-(2**31), 2**31 - 1),
    pl.UInt32: (0, 2**32 - 1),
}


def sum_without_wrapping(expression: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    """The sum of ``expression``, worked out as Int64 when its type would wrap past its limit."""
    return expression.cast(pl.Int64).sum() if dtype in NARROW_SUM_LIMITS else expression.sum()


def totals_that_do_not_fit(frame: pl.DataFrame, column: str, dtype: pl.DataType) -> pl.DataFrame:
    """The rows of ``frame`` whose ``column`` lies outside what ``dtype`` can hold."""
    low, high = NARROW_SUM_LIMITS[dtype]
    return frame.filter(pl.col(column).is_not_null() & ~pl.col(column).is_between(low, high))


class GroupBy:
    """One row per distinct combination of ``by``, carrying the aggregations named in config."""

    name = "group_by"
    description = "Collapse `rows` to one row per distinct key, carrying only what is named."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "by": ConfigField(
                type="array",
                description="Column names forming the group key; a null value groups with other nulls.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="string",
                    description="One column name; naming the same column twice counts it once.",
                ),
            ),
            "aggregations": ConfigField(
                type="array",
                description="One element per output column beyond the key: how it is computed.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One aggregation: its output name, which aggregate it uses, and what it reads.",
                    fields={
                        "name": ConfigField(type="string", description="The output column's name.", required=True),
                        "function": ConfigField(
                            type="string",
                            description="Which aggregate this element computes, drawn from the aggregates that ship.",
                            required=True,
                            enum=tuple(sorted(AGGREGATE_FUNCTIONS)),
                        ),
                        "value": ConfigField(
                            type="expression",
                            description="What is aggregated; omit it for 'count', which counts rows "
                            "and reads no column at all.",
                            required=False,
                        ),
                        "separator": ConfigField(
                            type="string",
                            description="What goes between the values 'join_text' joins, defaulting "
                            "to a comma and a space; only 'join_text' reads it, and giving it beside "
                            "any other aggregate is refused.",
                            required=False,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "One row per distinct key, in the order each key first appeared -- not sorted, "
                "because a key spanning several columns has no obvious total order. Every column "
                "neither named in 'by' nor produced by an aggregation disappears; 'min' and 'max' "
                "are the only way to carry one forward unchanged. A null key groups with other "
                "nulls rather than being dropped, matching what this tool is named after."
            ),
            examples=(_total_and_count_example(), _carried_column_example(), _joined_text_example()),
            notes=(
                "'sum' of an all-null group is null, not zero: zero is a measured amount and null "
                "is the absence of one, and nothing downstream can tell a fabricated zero from a "
                "real one. Nulls are still skipped within a group holding any value at all.",
                "'first' reads the order rows arrived in, and is the one aggregate here that can "
                "answer differently for the same data. Use it for a column that is the same on every "
                "row of a group -- and count the distinct values beside it if that is an assumption "
                "rather than a certainty.",
                "'count' counts every row of a group and reads no column; 'count_non_null' reads one "
                "and ignores its nulls. Porting a step from a library that spells the non-null count "
                "'count' would otherwise silently swap the two numbers.",
                "'join_text' joins a group's text in the order the rows arrived, separated by "
                "'separator' or by a comma and a space where none is given, and skips the values "
                "that are missing. A group whose values are all missing joins to the empty string, "
                "which is the opposite of the ruling 'sum' makes about the same group: joining no "
                "strings produces the empty string, while adding no numbers produces a total nobody "
                "measured. What is joined has to be text, and a separator beside any other "
                "aggregate is refused rather than ignored.",
                "A 'value' that reads no column, such as '1', is refused. A fixed value is worked "
                "out once per group, not once per row, so 'sum' of 1 would not count the rows. Use "
                "'count' for that.",
                "A 'sum' keeps the type of what it adds up. A total of Int32 or UInt32 values that "
                "does not fit in that type stops the run, naming the aggregation, rather than "
                "wrapping round to a wrong number. Cast the column to Int64 first if totals can be "
                "that large.",
            ),
            field_help={
                "by": "The group key; repeating a column counts it once and does not duplicate it on the output.",
                "aggregations": "Every output column beyond the key comes from here; nothing else survives the group.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Every rule a template alone cannot express, answered from the configuration alone.

        None of these needs the frame in front of it: a name colliding with the key, two
        aggregations sharing a name, a row count given something to count, an aggregate with
        nothing to aggregate, and a separator with nothing to separate are all questions the
        configuration answers about itself.
        """
        by_columns = set(config["by"])
        seen_names: set[str] = set()
        for index, aggregation in enumerate(config["aggregations"]):
            name = aggregation["name"]
            function = aggregation["function"]
            value = aggregation.get("value")

            require_not_blank(name, f"aggregations[{index}].name")
            if name in by_columns:
                raise ConfigError(
                    f"aggregation output name {name!r} collides with group-by column {name!r}; "
                    f"give the aggregation a different name"
                )
            if name in seen_names:
                raise ConfigError(f"two aggregations share the output name {name!r}; each must be unique")
            seen_names.add(name)

            if aggregation.get("separator") is not None and function != _JOINING_FUNCTION:
                raise ConfigError(
                    f"config field 'aggregations[{index}].separator' sits beside function "
                    f"{function!r}, which has nothing to separate; only 'join_text' reads a "
                    f"separator, so either make the function 'join_text' or drop the separator"
                )

            if function == _NO_VALUE_FUNCTION:
                if value is not None:
                    raise ConfigError(
                        f"aggregation {name!r} uses 'count', which counts rows and takes no 'value'; "
                        f"use 'count_non_null' to count the values of a column instead"
                    )
            elif value is None:
                raise ConfigError(f"aggregation {name!r} declares function {function!r} but no 'value' to aggregate")
            elif not references(value):
                # polars aggregates a fixed value once per group, not once per row, so sum of 1
                # would give 1 for every group instead of its row count.
                raise ConfigError(
                    f"config field 'aggregations[{index}].value' holds `{value}`, which reads no column; "
                    f"{function!r} of a fixed value is worked out once per group, not once per row. To "
                    f"count rows use 'count'. To add up a fixed amount per row, write it as a column in "
                    f"a formula step before this one and aggregate that column"
                )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """The columns this configuration names, checked against the frame that actually arrived.

        Read off the schema alone, so a frame of a million rows and one of none are refused
        identically: a 'by' column that is not there, a 'value' expression naming one that is not,
        a 'sum' asked to add up something that is not a number, and a 'join_text' asked to join
        something that is not text. The two dtype rulings need the schema because an empty frame
        would otherwise sum nothing and join nothing, and raise neither.
        """
        frame = inputs["rows"]
        require_columns(frame, _by_columns(config), "by", "rows")

        for index, aggregation in enumerate(config["aggregations"]):
            value = aggregation.get("value")
            if value is None:
                continue
            field = f"aggregations[{index}].value"
            if aggregation["function"] == "sum":
                column, dtype = compiled_dtype(frame, value, field)
                if not dtype.is_numeric():
                    raise ConfigError(
                        f"config field {field!r} sums column {column!r}, which holds {dtype}, not a "
                        f"number, in expression: `{value}`"
                    )
            elif aggregation["function"] == _JOINING_FUNCTION:
                column, dtype = compiled_dtype(frame, value, field)
                if dtype != pl.String:
                    raise ConfigError(
                        f"config field {field!r} asks 'join_text' to join column {column!r}, which "
                        f"holds {dtype}, not text, in expression: `{value}`; format it into text in "
                        f"the step before this one, where the formatting is written down"
                    )
            else:
                compile_for(frame, value, field)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        by_columns = _by_columns(config)
        aggregates = [
            _compile_aggregation(frame, index, aggregation) for index, aggregation in enumerate(config["aggregations"])
        ]
        grouped = frame.group_by(by_columns, maintain_order=True).agg(aggregates)
        return Ports({"rows": _narrow_totals_back(frame, grouped, config)})


def _by_columns(config: Mapping[str, Any]) -> list[str]:
    """The group key, a column named twice counted once, in the order it was first written."""
    return list(dict.fromkeys(config["by"]))


def _narrow_totals_back(frame: pl.DataFrame, grouped: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """Give each sum of a 32-bit column its own type back, or stop if a total does not fit in it."""
    for index, aggregation in enumerate(config["aggregations"]):
        if aggregation["function"] != "sum":
            continue
        field = f"aggregations[{index}].value"
        _, dtype = compiled_dtype(frame, aggregation["value"], field)
        if dtype not in NARROW_SUM_LIMITS:
            continue
        name = aggregation["name"]
        low, high = NARROW_SUM_LIMITS[dtype]
        offending = totals_that_do_not_fit(grouped, name, dtype)
        if offending.height:
            raise DataError(
                f"aggregation {name!r} ({field}) adds up {dtype} values, and the total of some "
                f"groups does not fit in {dtype}, which holds {low:,} to {high:,}; it would have "
                f"wrapped round to a wrong number. Cast the column to Int64 with cast_columns before "
                f"this step. The totals shown are the true ones",
                offending=offending,
                total=grouped.height,
            )
        grouped = grouped.with_columns(pl.col(name).cast(dtype))
    return grouped


def _compile_aggregation(frame: pl.DataFrame, index: int, aggregation: Mapping[str, Any]) -> pl.Expr:
    """One aggregation's expression, named for the column it produces."""
    name = aggregation["name"]
    function = aggregation["function"]
    value = aggregation.get("value")

    if function == "count":
        return pl.len().cast(pl.Int64).alias(name)

    expression = compile_for(frame, value, f"aggregations[{index}].value")

    if function == "sum":
        _, dtype = compiled_dtype(frame, value, f"aggregations[{index}].value")
        total = sum_without_wrapping(expression, dtype)
        return pl.when(expression.is_not_null().any()).then(total).otherwise(None).alias(name)
    if function == "count_non_null":
        return expression.count().cast(pl.Int64).alias(name)
    if function == "n_unique":
        return expression.n_unique().cast(pl.Int64).alias(name)
    if function == "min":
        return expression.min().alias(name)
    if function == "first":
        return expression.first().alias(name)
    if function == _JOINING_FUNCTION:
        separator = aggregation.get("separator")
        separator = DEFAULT_SEPARATOR if separator is None else separator
        return expression.str.join(separator).alias(name)
    if function == "max":
        return expression.max().alias(name)
    # Every published aggregate has a branch of its own, and this line is what makes that a fact
    # rather than a claim. A dispatcher that ends by returning one of the aggregates answers a name
    # it does not recognise with a plausible number under the wrong name -- CrossTab's ``first``,
    # computed as ``max`` because it had no branch, is that failure as it actually happened.
    # ``validate_config`` has already whitelisted the name against AGGREGATE_FUNCTIONS, so reaching
    # here means the set grew and this function did not.
    raise AssertionError(f"unreachable aggregate function {function!r}")  # pragma: no cover


def _total_and_count_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A"],
                    "amount": [10.0, 20.0, 5.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B"],
                    "total": [15.0, 20.0],
                    "lines": [2, 1],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a per-group total and a row count, in first-appearance order",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "total", "function": "sum", "value": "[amount]"},
                {"name": "lines", "function": "count"},
            ],
        },
        explanation="'code' groups the rows in the order 'A' and 'B' first appear; 'total' adds "
        "'amount' within each group and 'lines' counts how many rows joined it.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _carried_column_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", None],
                    "region": ["x", "y", "x", "z"],
                    "amount": [10.0, None, 5.0, 3.0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", None],
                    "total": [15.0, None, 3.0],
                    "region_of_group": ["x", "y", "z"],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a null key is its own group, and 'min' carries a constant column forward",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "total", "function": "sum", "value": "[amount]"},
                {"name": "region_of_group", "function": "min", "value": "[region]"},
            ],
        },
        explanation="The row with no 'code' is a group of its own rather than being dropped; group "
        "'B' sums to null because its only 'amount' is missing. 'region' is constant within every "
        "group here, and 'min' is how it survives even though 'region' is neither a key nor an "
        "ordinary aggregate.",
        inputs=inputs,
        outputs=outputs,
    )


def _joined_text_example() -> WorkedExample:
    inputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "A", "B", "C"],
                    "note": ["late", "checked", None, "held", None],
                }
            )
        }
    )
    outputs = Ports(
        {
            "rows": pl.DataFrame(
                {
                    "code": ["A", "B", "C"],
                    "notes": ["late", "checked; held", ""],
                    "lines": [2, 2, 1],
                }
            )
        }
    )
    return WorkedExample(
        scenario="a group's notes kept as one string, including the group that said nothing",
        configuration={
            "by": ["code"],
            "aggregations": [
                {"name": "notes", "function": "join_text", "value": "[note]", "separator": "; "},
                {"name": "lines", "function": "count"},
            ],
        },
        explanation="'join_text' joins each group's notes in the order the rows arrived, with the "
        "separator written down rather than assumed. Group A's missing note is skipped instead of "
        "joined in as a blank, which is why 'notes' reads 'late' while 'lines' still counts two "
        "rows -- the count beside the join is how a reader tells one note from one note and a gap. "
        "Group C had a row and no note on it, so there was nothing to join and the answer is the "
        "empty string, not null. That is deliberately the opposite of what 'sum' says about the "
        "same group: joining no strings together produces a string, while adding no numbers "
        "together produces a total nobody measured.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(GroupBy())
