"""Join: two input ports, three output ports, and no join type between them.

**There is no `join_type`.** Inner, left, right and outer are not four behaviours a configuration
picks between; they are four ways of reading the same pairing. So this tool always does the pairing
once and hands back all three of its results on named ports -- `matched` for the pairs, `left_only`
and `right_only` for the rows of each side that found nobody -- and which join a flow performed is
decided by which of those ports it wires. A left join reads `matched` and `left_only`; an outer join
reads all three; an anti-join reads `left_only` alone. What that buys is the orphan check for free:
the rows a pairing failed to account for are an edge a flow can route to a report or an assertion,
rather than a number someone reconstructs afterwards with a second join they have to remember to
write.

**Fan-out is real, and it is not a defect.** A key that appears twice on the right turns one left row
into two output rows, so `matched.height` is unrelated to either input's height, and a fanned-out
amount is *repeated* rather than apportioned. Any other behaviour would require this tool to pick
one of the matches, and nothing in its configuration gives it a basis to pick. Uniqueness of a key
is the caller's business, guarded upstream -- with RemoveDuplicates, or with an assertion -- and the
reason the repetition is left visible is that a reader has to see it before deciding whether their
downstream total is legitimate.

**A null key never pairs**, so such a row leaves on its own side's orphan port. This is deliberately
the *opposite* of the ruling RemoveDuplicates makes on the same-looking question, and both are
right: SQL's `DISTINCT` treats two nulls as one group, SQL's `JOIN` treats them as unpairable. A
missing key is an absence of information, and pairing two absences would invent a relationship out
of nothing -- worse, on dirty data it would fan every null-keyed row on one side across every
null-keyed row on the other, turning a small missing number into a large wrong one.

**Row accounting is stated over rows, not counts.** Fan-out breaks addition, so
`matched.height + left_only.height == left.height` is simply false and is not the promise. The
promise is that every row of a side is either in `matched` at least once or in its own orphan port
exactly once, never both and never neither.

**Row order is deterministic and nothing more.** `matched` comes back in left-input order, with one
left row's fan-out copies in right-input order; each orphan port comes back in its own side's input
order. That costs an explicit order-maintaining join, and the parallelism it gives up, and it is
worth paying because a flow that writes a file needs a diff that does not churn between runs. It is
not a licence to depend on order for correctness: a step that needs a meaningful order still needs a
Sort in front of it.

**A collision the suffix cannot settle is refused here, off the schema, before any row moves.** Left
to the query engine it arrives as an error about a duplicate column, from a library the author has
never heard of, naming neither the field to change nor the fact that changing `suffix` is the fix.
"""

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
    require_columns,
)

# The key fields, in the order a refusal quotes them. Named once so the cross-field rule below and
# the messages it raises cannot drift apart from the template.
_KEY_FIELDS: Final[tuple[str, str, str]] = ("keys", "left_on", "right_on")


_HOW_ENUM: Final[tuple[str, ...]] = ("inner", "left")


class Join:
    """Pair `left` to `right` on a key, and hand back the pairs and both sides' orphans."""

    name = "join"
    description = "Pair `left` to `right` on a key, reporting the pairs and each side's unpaired rows."
    category = "combine"
    input_ports = ("left", "right")
    output_ports = ("matched", "left_only", "right_only")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "keys": ConfigField(
                type="array",
                description="Key columns spelled the same on both sides; give this or both of "
                "'left_on' and 'right_on', never both forms.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column name, present on both sides."),
            ),
            "left_on": ConfigField(
                type="array",
                description="The left side's key columns, when the two sides spell the key "
                "differently; requires 'right_on' of the same length.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column name on the left side."),
            ),
            "right_on": ConfigField(
                type="array",
                description="The right side's key columns, paired position by position with "
                "'left_on'; requires 'left_on' of the same length.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column name on the right side."),
            ),
            "suffix": ConfigField(
                type="string",
                description="What a right-hand non-key column takes when its name collides with a "
                "left-hand one; key columns never double up.",
                default="_right",
            ),
            "how": ConfigField(
                type="string",
                description="Whether `matched` carries only the rows that paired ('inner') or every "
                "left row, with the right-hand columns null where none paired ('left').",
                default="inner",
                enum=_HOW_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "One pairing, three ports: `matched` carries a row per pair, so a key repeated on the "
                "right fans one left row into a copy per match, and `left_only` / `right_only` carry "
                "the rows of each side that paired with nothing -- which is why there is no join-type "
                "option to choose between. An orphan port carries its own side's columns exactly as "
                "they arrived: no suffix, no columns from the other side, because an unpaired row is "
                "an input row, not a join result with nulls in it."
            ),
            examples=(_fan_out_example(), _differently_named_key_example()),
            notes=(
                "`how` decides only which rows reach `matched`; the orphan ports answer their own "
                "question either way. Under 'left' an unmatched left row appears on both -- carried "
                "in `matched` with its right-hand columns null, and reported on `left_only` -- "
                "because one port is the result and the other is the diagnostic.",
                "A right join is this join with the two inputs swapped, which is why 'right' is not "
                "offered: the same result under two spellings is a thing to get wrong.",
                "A null key never pairs, so a row whose key is missing leaves on its own side's "
                "orphan port. This is the opposite of RemoveDuplicates, where two nulls are one "
                "group: sameness there follows DISTINCT, sameness here follows JOIN. Reading the two "
                "tools side by side, that difference is deliberate in both.",
                "Fan-out is not a defect: a fanned-out amount is repeated on every copy, never "
                "apportioned between them, so a total taken downstream of a join is only meaningful "
                "if the key is unique on the side being joined to. Guard that upstream.",
                "Row order is deterministic -- left-input order, with one left row's copies in "
                "right-input order, and each orphan port in its own side's order -- so a flow that "
                "writes files gets diffs that do not churn. It is not a semantic guarantee: a step "
                "that needs a meaningful order still needs a Sort in front of it.",
            ),
            field_help={
                "how": "'inner' keeps only the rows that paired. 'left' keeps every left row, "
                "nulling the right-hand columns where nothing paired -- the commonest join a flow "
                "asks for. The column set is identical either way, so a later step naming a column "
                "does not depend on which was chosen.",
                "keys": "Key columns shared by both sides; use 'left_on'/'right_on' instead when the "
                "spellings differ, never both forms at once.",
                "left_on": "The left spelling of each key column, and the spelling that survives on "
                "'matched'; must be the same length as 'right_on'.",
                "right_on": "The right spelling of each key column, paired by position with "
                "'left_on'; the right key column itself does not appear on 'matched' at all.",
                "suffix": "Applied only to a colliding right-hand non-key column; if the suffixed "
                "name is taken as well, the join is refused rather than guessing another suffix.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """The one rule a per-field template cannot express: which key fields may be given together.

        Each of the three key fields is individually optional and individually valid, so only the
        *combination* can be wrong. The configuration arrives with its defaults already filled, so
        every template key is present and a membership test is always true: ``config["keys"] is
        None`` is the only spelling that asks whether a field was given.
        """
        keys, left_on, right_on = (config[field] for field in _KEY_FIELDS)

        if keys is not None:
            also_given = [field for field in ("left_on", "right_on") if config[field] is not None]
            if also_given:
                raise ConfigError(
                    f"config gives 'keys' together with {', '.join(repr(field) for field in also_given)}; "
                    f"give either 'keys', or both of 'left_on' and 'right_on', never both forms"
                )
            return None

        if left_on is None and right_on is None:
            raise ConfigError(
                "config names no key columns: give 'keys' for a key spelled the same on both sides, or "
                "both of 'left_on' and 'right_on' when the two sides spell it differently"
            )
        if left_on is None or right_on is None:
            given, missing = ("left_on", "right_on") if right_on is None else ("right_on", "left_on")
            raise ConfigError(
                f"config gives {given!r} without {missing!r}; the two-field form needs both halves, or "
                f"give 'keys' instead when both sides spell the key the same way"
            )
        if len(left_on) != len(right_on):
            raise ConfigError(
                f"config fields 'left_on' and 'right_on' must name the same number of key columns, "
                f"since they pair position by position; 'left_on' names {len(left_on)} "
                f"({', '.join(left_on)}) and 'right_on' names {len(right_on)} ({', '.join(right_on)})"
            )
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """The two rules that need the schemas: keys where promised, and no unsettled collision.

        Both are read off the schemas alone, so a configuration that cannot work is refused whether
        the frames carry a million rows or none at all -- and refused as a configuration problem
        rather than as a tool that failed while running, which is what an author can act on.

        Each key is checked against the schema of *its own* side. A column that is on the right and
        not on the left would pass a check that merely looked for the name somewhere, and fail
        inside the query engine instead; a two-input tool gives an author two places to look, so
        each refusal says which one. The collision rule is the one check here that no single-input
        tool could ever need: it is a question about a *pair* of schemas together.
        """
        left, right = inputs["left"], inputs["right"]
        left_keys, right_keys = _key_columns(config)
        shared = config["keys"] is not None

        require_columns(left, left_keys, "keys" if shared else "left_on", "left")
        require_columns(right, right_keys, "keys" if shared else "right_on", "right")

        _refuse_unsettled_collision(
            left.columns,
            _matched_right_names(left.columns, right.columns, right_keys, config["suffix"]),
        )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        left, right = inputs["left"], inputs["right"]
        suffix = config["suffix"]
        left_keys, right_keys = _key_columns(config)
        right_names = _matched_right_names(left.columns, right.columns, right_keys, suffix)

        # `maintain_order` is what turns three query-engine artefacts into the documented order; the
        # inputs themselves are never touched, so a frame fanning out to two branches is safe.
        # 'left' keeps every left row and nulls the right-hand columns where nothing paired, which
        # is the commonest join a flow asks for and the one this tool used to make a reader build out
        # of two steps. The column set and its order are the same either way, so the two modes differ
        # in which rows arrive and never in what a downstream step may name.
        matched = left.join(
            right,
            left_on=left_keys,
            right_on=right_keys,
            how="inner" if config["how"] == "inner" else "left",
            suffix=suffix,
            nulls_equal=False,
            maintain_order="left_right",
        ).select(*left.columns, *right_names.values())

        # An orphan port is an anti-join rather than a left join filtered on a null, because a
        # matched row may legitimately hold a null in a non-key column and the two are not the same
        # question. Only the key columns of the other side are read: nothing else can affect pairing.
        left_only = left.join(
            right.select(right_keys),
            left_on=left_keys,
            right_on=right_keys,
            how="anti",
            nulls_equal=False,
            maintain_order="left",
        )
        right_only = right.join(
            left.select(left_keys),
            left_on=right_keys,
            right_on=left_keys,
            how="anti",
            nulls_equal=False,
            maintain_order="left",
        )

        return Ports({"matched": matched, "left_only": left_only, "right_only": right_only})


def _key_columns(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Each side's key columns, whether the configuration spelled the key once or twice.

    One reading of the two valid forms, used by both the check and the work, so that the shared
    spelling cannot be resolved one way before the run and another way during it. The cross-field
    rule has already refused everything except those two forms by the time this is reached.
    """
    shared = config["keys"]
    if shared is not None:
        return list(shared), list(shared)
    return list(config["left_on"]), list(config["right_on"])


def _matched_right_names(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    right_keys: Sequence[str],
    suffix: str,
) -> dict[str, str]:
    """Every right-hand column that reaches `matched`, mapped to the name it arrives under.

    The key columns are absent by design: a pair of key columns is one column after the join, and
    keeping both would leave `matched` carrying two columns of identical values that every
    downstream step would then have to choose between. Under `left_on`/`right_on` that means the
    left spelling is the one that survives, which is not something a reader can guess.
    """
    return {
        name: (f"{name}{suffix}" if name in left_columns else name) for name in right_columns if name not in right_keys
    }


def _refuse_unsettled_collision(left_columns: Sequence[str], right_names: Mapping[str, str]) -> None:
    """Refuse a collision the suffix cannot settle, in this library's own words.

    Reached when the *suffixed* name is taken too. The query engine would refuse it as well, but as
    an error about a duplicate column that names neither this tool's configuration nor the fact that
    a different `suffix` is the fix -- and it would only do so once rows were already moving.
    """
    taken = set(left_columns)
    for source, target in right_names.items():
        if target in taken:
            raise ConfigError(
                f"joining would produce two columns named {target!r}: the right input's {source!r} "
                f"cannot be renamed out of the way, because {target!r} is already taken. Give config "
                f"field 'suffix' a value that leads to a name no column carries yet."
            )
        taken.add(target)


def _fan_out_example() -> WorkedExample:
    inputs = Ports(
        {
            "left": pl.DataFrame(
                {
                    "entry_id": [1, 2, 3],
                    "group_id": ["G1", "G1", "G4"],
                    "amount": [10.0, 20.0, 40.0],
                }
            ),
            "right": pl.DataFrame(
                {
                    "header_id": [10, 11, 12],
                    "group_id": ["G1", "G1", "G3"],
                    "state": ["open", "closed", "open"],
                }
            ),
        }
    )
    outputs = Ports(
        {
            "matched": pl.DataFrame(
                {
                    "entry_id": [1, 1, 2, 2],
                    "group_id": ["G1", "G1", "G1", "G1"],
                    "amount": [10.0, 10.0, 20.0, 20.0],
                    "header_id": [10, 11, 10, 11],
                    "state": ["open", "closed", "open", "closed"],
                }
            ),
            "left_only": pl.DataFrame({"entry_id": [3], "group_id": ["G4"], "amount": [40.0]}),
            "right_only": pl.DataFrame({"header_id": [12], "group_id": ["G3"], "state": ["open"]}),
        }
    )
    return WorkedExample(
        scenario="pair rows to a group whose key the other side carries twice",
        configuration={"keys": ["group_id"]},
        explanation="Group 'G1' is on the right twice, so each of the two left rows carrying it fans "
        "into two matched rows -- four rows out of three in, with each amount repeated rather than "
        "split. Group 'G4' pairs with nothing and leaves on 'left_only'; 'G3' pairs with nothing on "
        "the other side and leaves on 'right_only'. The row order shown is the order produced: left "
        "order, and right order within one left row's copies.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


def _differently_named_key_example() -> WorkedExample:
    inputs = Ports(
        {
            "left": pl.DataFrame(
                {
                    "movement_id": [71, 72],
                    "period_start": ["2024-01", "2024-03"],
                    "amount": [100.0, 300.0],
                }
            ),
            "right": pl.DataFrame(
                {
                    "netting_id": [81, 82],
                    "period": ["2024-01", "2024-02"],
                    "amount": [-40.0, -20.0],
                }
            ),
        }
    )
    outputs = Ports(
        {
            "matched": pl.DataFrame(
                {
                    "movement_id": [71],
                    "period_start": ["2024-01"],
                    "amount": [100.0],
                    "netting_id": [81],
                    "amount_right": [-40.0],
                }
            ),
            "left_only": pl.DataFrame({"movement_id": [72], "period_start": ["2024-03"], "amount": [300.0]}),
            "right_only": pl.DataFrame({"netting_id": [82], "period": ["2024-02"], "amount": [-20.0]}),
        }
    )
    return WorkedExample(
        scenario="pair two sides whose key column is spelled differently, over a colliding amount",
        configuration={"left_on": ["period_start"], "right_on": ["period"]},
        explanation="Only the left spelling 'period_start' survives on 'matched'; the right's "
        "'period' does not appear at all, suffixed or otherwise, because the pair is one column "
        "after the join. 'amount' is on both sides and is not a key, so the right's value survives "
        "under the default suffix as 'amount_right'. Each orphan port keeps its own side's own "
        "spelling, including the 'period' the join dropped.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Join())
