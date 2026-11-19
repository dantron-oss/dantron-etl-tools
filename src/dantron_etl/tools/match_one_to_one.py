"""MatchOneToOne: take the best pairs first, and never use a key twice.

Each row of ``candidates`` is one possible pairing of a left key with a right key. The tool walks the
candidates from best to worst and accepts a pair when neither of its keys has been used yet. A join
and a filter cannot say this, because whether a candidate is still open depends on which candidates
were accepted before it.

**The order of the walk.** The configured ``priority`` columns come first, each in its own direction,
like ``sort``. A direction left out means ascending. Missing values sort last. The tool then adds the
left key ascending and the right key ascending as the last two tie-breaks. So two candidates are
never in an undecided order, and the result does not depend on the order the rows arrived in. Both
output ports come out in this order.

**Two output ports.** ``accepted`` holds the pairs taken. ``rejected`` holds every other candidate.
Every input row leaves on exactly one of them, with all its columns, like ``filter``.

**Ties.** With ``ties: first`` (the default), the walk order decides: candidates equal on every
priority column fall to the smaller left key, then the smaller right key.

With ``ties: refuse``, the walk stops at each candidate whose two keys are both free and looks for a
rival. A rival is another candidate that shares the left key (or the right key), has exactly the
same values on every priority column, and is still available. Available means neither of its keys
has been taken or refused yet. Two missing values count as the same value here. The key tie-break
never counts as a difference. A key with a rival is refused. It stays unmatched for the rest of the
walk, and every candidate naming it is rejected. A refusal never takes the key on the other side of
the pair, so that key stays free for other candidates.

**A missing key pairs with nothing**, as in ``join``, which follows SQL's ``=``. A candidate with a
missing left or right key goes to ``rejected`` and takes no key from anyone.

**How the walk runs on whole columns.** It goes in rounds. In each round, every open candidate whose
outcome no earlier open candidate can still change is decided at once. Under ``first`` that is a
candidate that comes first for both its keys. Under ``refuse`` it must also come before every open
candidate that could change whether its rivals are still available. The best open candidate always
qualifies, so each round decides at least one. The number of rounds grows with the longest chain of
candidates pushed to a second choice, not with the number of rows.
"""

from collections.abc import Mapping
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    require_columns,
)

_DIRECTION_ENUM: Final[tuple[str, ...]] = ("asc", "desc")
_TIES_ENUM: Final[tuple[str, ...]] = ("first", "refuse")

# A default inside an array element is never filled by the contract, so the tool applies it itself.
_DEFAULT_DIRECTION: Final[str] = "asc"


class MatchOneToOne:
    """Accept the best candidate pairs first, so no left key and no right key is matched twice."""

    name = "match_one_to_one"
    description = "Pair left keys with right keys one to one, taking the best candidates first."
    category = "preparation"
    input_ports = ("candidates",)
    output_ports = ("accepted", "rejected")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "left_key": ConfigField(
                type="string",
                description="The column holding the left key of each candidate pair.",
                required=True,
                non_empty=True,
            ),
            "right_key": ConfigField(
                type="string",
                description="The column holding the right key of each candidate pair.",
                required=True,
                non_empty=True,
            ),
            "priority": ConfigField(
                type="array",
                description="The columns that rank the candidates, most important first. The best "
                "candidate is taken first.",
                required=True,
                non_empty=True,
                items=ConfigField(
                    type="object",
                    description="One ranking column, and which way it reads.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column to rank on. It has to be on the input.",
                            required=True,
                            non_empty=True,
                        ),
                        "direction": ConfigField(
                            type="string",
                            description="'asc' puts the smallest value first, 'desc' the largest. "
                            "Left out, it is 'asc'.",
                            enum=_DIRECTION_ENUM,
                        ),
                    },
                ),
            ),
            "ties": ConfigField(
                type="string",
                description="What to do when a key's best candidates are equal on every priority "
                "column: 'first' takes the smaller key, 'refuse' leaves the key unmatched.",
                default="first",
                enum=_TIES_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Each row is one possible pair of a left key and a right key. The tool walks the rows "
                "from best to worst by 'priority', then by left key, then by right key. It accepts a "
                "pair when neither key has been used. Accepted pairs go to 'accepted'. Every other "
                "row goes to 'rejected'. Each row leaves on exactly one port, with all its columns, "
                "and both ports come out in the walk order."
            ),
            examples=(_first_example(), _refuse_example()),
            notes=(
                "A missing left or right key pairs with nothing, as a null key does in a SQL join. "
                "That row goes to 'rejected' and uses no key.",
                "Under 'refuse', a rival is another candidate that shares a key, has the same values "
                "on every priority column, and has neither key taken or refused yet. Two missing "
                "values count as the same. A key with a rival is refused and stays unmatched. The "
                "key on the other side stays free.",
                "Under 'refuse', two rows with the same keys and the same priority values are rivals "
                "of each other. Both keys are refused. Remove repeated rows first if they are copies.",
                "Missing values in a priority column sort last, in either direction.",
                "Two rows with the same keys and the same priority values come out in the order they "
                "arrived. Nothing else tells them apart.",
            ),
            field_help={
                "left_key": "The left key. Each left key is accepted at most once.",
                "right_key": "The right key. Each right key is accepted at most once. Must be a "
                "different column from 'left_key'.",
                "priority": "A list of ranking columns, each with an optional 'direction'. Later "
                "columns only break ties left by earlier ones.",
                "ties": "'first' (the default) or 'refuse'.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The rules a template cannot state: one column on both sides, and the same checks as sort.

        Like sort, a priority entry with a blank column, a direction given with no value, or a
        column named twice is refused, by its path.
        """
        if config["left_key"] == config["right_key"]:
            raise ConfigError(
                f"config fields 'left_key' and 'right_key' both name {config['left_key']!r}; a pair needs a left "
                f"key column and a different right key column"
            )
        seen: dict[str, int] = {}
        for index, key in enumerate(config["priority"]):
            path = f"priority[{index}]"
            column = key["column"]
            if not column.strip():
                raise ConfigError(f"config field '{path}.column' is blank; a priority entry needs a column name")
            if "direction" in key and key["direction"] is None:
                raise ConfigError(
                    f"config field '{path}.direction' was given with no value; write 'asc' or 'desc', "
                    f"or leave it out to take 'asc'"
                )
            if column in seen:
                raise ConfigError(
                    f"config field 'priority' names column {column!r} twice, at {path} and at "
                    f"priority[{seen[column]}]; the second one can never decide anything"
                )
            seen[column] = index

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Both key columns and every priority column have to be on the input."""
        frame = inputs["candidates"]
        require_columns(frame, [config["left_key"]], "left_key", "candidates")
        require_columns(frame, [config["right_key"]], "right_key", "candidates")
        require_columns(frame, [key["column"] for key in config["priority"]], "priority", "candidates")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Sort into walk order, decide the candidates in rounds, then split the rows by outcome."""
        frame = inputs["candidates"]
        left, right = config["left_key"], config["right_key"]
        columns = [key["column"] for key in config["priority"]]
        descending = [key.get("direction", _DEFAULT_DIRECTION) == "desc" for key in config["priority"]]

        order = frame.select(
            pl.arg_sort_by(
                [*columns, left, right],
                descending=[*descending, False, False],
                nulls_last=True,
                maintain_order=True,
            )
        ).to_series()
        ordered = frame[order]

        # A tie group is a run of candidates equal on every priority column. The frame is sorted on
        # those columns first, so equal values sit next to each other.
        changed = pl.any_horizontal([pl.col(column).ne_missing(pl.col(column).shift(1)) for column in columns])
        work = ordered.select(
            pl.int_range(pl.len(), dtype=pl.get_index_type()).alias("rank"),
            pl.col(left).alias("l"),
            pl.col(right).alias("r"),
            changed.fill_null(True).cum_sum().alias("tie"),
        ).filter(pl.col("l").is_not_null() & pl.col("r").is_not_null())

        refuse = config["ties"] == "refuse"
        accepted: list[pl.Series] = []
        # The walk runs in rounds. `work` holds the open candidates: not yet decided, and both keys
        # still free. A round decides every open candidate that no earlier open candidate can affect.
        #
        # - 'first': a candidate's fate depends only on its own two keys. If no earlier open
        #   candidate shares either key, nothing can take them first, so it is accepted now.
        # - 'refuse': it also depends on whether its rivals are still open. A rival shares one of its
        #   keys. The rival can be closed by an earlier candidate sharing the rival's other key. So
        #   the candidate must come first among everything one key away, and everything one key
        #   away from those.
        #
        # Two candidates decided in the same round are never that close to each other, so deciding
        # them together gives the same answer as walking them one at a time. The best open candidate
        # is always decided, so every round removes at least one row and the loop ends.
        while work.height:
            # "near": the best rank among open candidates sharing a key with this one, itself
            # included. "reach": the same, taken one key further, for 'refuse'. Write "near" to a
            # column first. Taking one window over another inside a single expression gave wrong
            # answers. The regression test in the test file pins that case.
            rounds = work.with_columns(
                near=pl.min_horizontal(pl.col("rank").min().over("l"), pl.col("rank").min().over("r"))
            )
            if refuse:
                reach = pl.min_horizontal(pl.col("near").min().over("l"), pl.col("near").min().over("r"))
            else:
                reach = pl.col("near")
            rounds = rounds.with_columns(
                decided=pl.col("rank") == reach,
                left_rival=pl.len().over("l", "tie") > 1 if refuse else pl.lit(False),
                right_rival=pl.len().over("r", "tie") > 1 if refuse else pl.lit(False),
            ).filter("decided")
            rounds = rounds.with_columns(taken=~pl.col("left_rival") & ~pl.col("right_rival"))
            accepted.append(rounds.filter("taken")["rank"])
            used_left = rounds.filter(pl.col("taken") | pl.col("left_rival"))["l"]
            used_right = rounds.filter(pl.col("taken") | pl.col("right_rival"))["r"]
            work = work.filter(
                ~pl.col("rank").is_in(rounds["rank"].implode())
                & ~pl.col("l").is_in(used_left.implode())
                & ~pl.col("r").is_in(used_right.implode())
            )

        taken = pl.Series("rank", [], dtype=pl.get_index_type())
        if accepted:
            taken = pl.concat(accepted)
        mask = pl.int_range(ordered.height, dtype=pl.get_index_type(), eager=True).is_in(taken.implode())
        return Ports({"accepted": ordered.filter(mask), "rejected": ordered.filter(~mask)})


def _first_example() -> WorkedExample:
    inputs = Ports(
        {
            "candidates": pl.DataFrame(
                {
                    "left_key": ["A", "B", "B", "C"],
                    "right_key": ["X", "X", "Y", None],
                    "gap": [1, 2, 3, 0],
                }
            )
        }
    )
    outputs = Ports(
        {
            "accepted": pl.DataFrame({"left_key": ["A", "B"], "right_key": ["X", "Y"], "gap": [1, 3]}),
            "rejected": pl.DataFrame(
                {"left_key": ["C", "B"], "right_key": [None, "X"], "gap": [0, 2]},
                schema={"left_key": pl.String, "right_key": pl.String, "gap": pl.Int64},
            ),
        }
    )
    return WorkedExample(
        scenario="match each left key to a right key, smallest gap first",
        configuration={"left_key": "left_key", "right_key": "right_key", "priority": [{"column": "gap"}]},
        explanation="C has the smallest gap but no right key, so it pairs with nothing. A-X comes "
        "next and is taken. B-X is next, but X is used, so B falls to its second choice, Y.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _refuse_example() -> WorkedExample:
    inputs = Ports(
        {
            "candidates": pl.DataFrame(
                {
                    "left_key": ["A", "A", "B", "A"],
                    "right_key": ["X", "Y", "X", "Z"],
                    "gap": [1, 1, 2, 5],
                }
            )
        }
    )
    outputs = Ports(
        {
            "accepted": pl.DataFrame({"left_key": ["B"], "right_key": ["X"], "gap": [2]}),
            "rejected": pl.DataFrame({"left_key": ["A", "A", "A"], "right_key": ["X", "Y", "Z"], "gap": [1, 1, 5]}),
        }
    )
    return WorkedExample(
        scenario="leave a key unmatched when its two best candidates are equally good",
        configuration={
            "left_key": "left_key",
            "right_key": "right_key",
            "priority": [{"column": "gap", "direction": "asc"}],
            "ties": "refuse",
        },
        explanation="A-X and A-Y both have gap 1, so A has no single best match and is refused. A-Z "
        "is rejected too, because A stays unmatched. X is not refused, so B takes it.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(MatchOneToOne())
