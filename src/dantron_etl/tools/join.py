"""Join: pair the rows of two tables on a key.

One pairing gives three outputs: `joined` for the pairs, and `left_only` and `right_only` for the
rows of each side that found no partner. A null key never pairs, as in SQL. A key repeated on the
right gives a left row one copy per match.

Why: docs/design/tools.md#join
"""

from collections.abc import Mapping, Sequence
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
from ..column_types import EXACT_TYPE, exact_dtype

# The key fields, in the order a refusal quotes them.
_KEY_FIELDS: Final[tuple[str, str, str]] = ("keys", "left_keys", "right_keys")


_HOW_ENUM: Final[tuple[str, ...]] = ("inner", "left")


class Join:
    """Pair `left` to `right` on a key, and hand back the pairs and both sides' unpaired rows."""

    name = "join"
    description = "Match the rows of two tables on a key column, and keep the rows that found no match."
    category = "combine"
    input_ports = ("left", "right")
    output_ports = ("joined", "left_only", "right_only")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "keys": ConfigField(
                type="array",
                description="The key columns, when both sides use the same names for them.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column, found on both sides."),
            ),
            "left_keys": ConfigField(
                type="array",
                description="The left side's key columns, when the two sides name the key differently.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column on the left side."),
            ),
            "right_keys": ConfigField(
                type="array",
                description="The right side's key columns, matched in order with 'left_keys'.",
                required=False,
                non_empty=True,
                items=ConfigField(type="string", description="One key column on the right side."),
            ),
            "suffix": ConfigField(
                type="string",
                description="Added to a right-hand column whose name is already used on the left.",
                default="_right",
            ),
            "how": ConfigField(
                type="string",
                description="Which rows go to `joined`: only the pairs ('inner'), or every left row ('left').",
                default="inner",
                enum=_HOW_ENUM,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use join to bring columns from one table onto the rows of another, matching on a key "
                "such as a customer id. `joined` has one row for each pair: the left row's columns, "
                "then the right row's. `left_only` and `right_only` hold the rows of each side that "
                "found no partner, with their own columns exactly as they came in. "
                "For an outer join read all three ports; for an anti join read `left_only`; "
                "for a right join swap the two inputs."
            ),
            examples=(_fan_out_example(), _differently_named_key_example()),
            notes=(
                "'how' only decides which rows go to `joined`. With 'left', a left row that found no "
                "partner is on `joined` with empty right-hand columns, and also on `left_only`.",
                "There is no 'right' option: a right join is this join with the inputs swapped.",
                "A row whose key is null never pairs, as with SQL's JOIN, so it goes to its own side's unpaired port.",
                "remove_duplicates treats null keys the opposite way: two nulls count as the same value "
                "there, as with SQL's DISTINCT.",
                "A key that appears twice on the right gives each matching left row two copies.",
                "A copied amount is repeated on each copy, never split, so a total after a join is "
                "right only if the key is unique on the right. Check that before the join.",
                "Rows come out in a fixed order: left order, a left row's copies in right order, and "
                "each unpaired port in its own side's order.",
                "That order keeps written files stable between runs; a step that needs a meaningful "
                "order still needs a sort.",
                "The two columns of a key pair must hold comparable types. Whole numbers of different "
                "sizes pair; a whole number against a Float64, or a date against a timestamp, is "
                "refused before any row moves. Use cast_columns first.",
                "Exact decimal (Decimal) keys pair by value, as SQL's = does: 1.50 pairs with 1.5000, "
                "and 5.00 pairs with the whole number 5. The key on `joined` keeps the left side's type.",
                "A Decimal key against a Float64 key is refused, naming the Float64 column. Convert one "
                "side with cast_columns first, so both are Decimal or both are Float64.",
                "A key column of type Null, as an empty branch gives, takes the other side's type, "
                "and none of its rows pair.",
                "Float64 keys pair only when exactly equal: 0.1 + 0.2 and 0.3 look alike but do not "
                "pair. Round a calculated key, or cast it to text or Decimal, first.",
            ),
            field_help={
                "how": "Write 'left' to keep every left row, which is the usual choice. Either way "
                "`joined` has the same columns, so later steps work with both.",
                "keys": "For example ['customer_id']. Give either 'keys', or both 'left_keys' and "
                "'right_keys', never both.",
                "left_keys": "For example ['period_start']. These names are the ones kept on `joined`. "
                "Give the same number of columns as 'right_keys'.",
                "right_keys": "For example ['period']. Matched by position with 'left_keys'. These "
                "columns are dropped from `joined`, since they repeat the left ones.",
                "suffix": "For example '_order'. Key columns never get it. If the new name is taken "
                "too, the step is refused.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse a configuration that gives the key both ways, neither way, or half of the two-field way.

        The defaults are already filled in, so `config["keys"] is None` is how to ask whether a field was given.
        """
        keys, left_keys, right_keys = (config[field] for field in _KEY_FIELDS)

        if keys is not None:
            also_given = [field for field in ("left_keys", "right_keys") if config[field] is not None]
            if also_given:
                raise ConfigError(
                    f"config gives 'keys' together with {', '.join(repr(field) for field in also_given)}; "
                    f"give either 'keys', or both of 'left_keys' and 'right_keys', never both forms"
                )
            return None

        if left_keys is None and right_keys is None:
            raise ConfigError(
                "config names no key columns: give 'keys' for a key spelled the same on both sides, or "
                "both of 'left_keys' and 'right_keys' when the two sides spell it differently"
            )
        if left_keys is None or right_keys is None:
            given, missing = ("left_keys", "right_keys") if right_keys is None else ("right_keys", "left_keys")
            raise ConfigError(
                f"config gives {given!r} without {missing!r}; the two-field form needs both halves, or "
                f"give 'keys' instead when both sides spell the key the same way"
            )
        if len(left_keys) != len(right_keys):
            raise ConfigError(
                f"config fields 'left_keys' and 'right_keys' must name the same number of key columns, "
                f"since they pair position by position; 'left_keys' names {len(left_keys)} "
                f"({', '.join(left_keys)}) and 'right_keys' names {len(right_keys)} ({', '.join(right_keys)})"
            )
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a missing key, a key pair that cannot be compared, or a clash of column names.

        All three are read from the column types, so an empty frame is refused the same way. Each key
        is checked on its own side, so the refusal says which side lacks it.
        """
        left, right = inputs["left"], inputs["right"]
        left_keys, right_keys = _key_columns(config)
        shared = config["keys"] is not None

        require_columns(left, left_keys, "keys" if shared else "left_keys", "left")
        require_columns(right, right_keys, "keys" if shared else "right_keys", "right")
        _refuse_keys_that_cannot_be_compared(left, right, left_keys, right_keys, shared)

        _refuse_unsettled_collision(
            left.columns,
            _joined_right_names(left.columns, right.columns, right_keys, config["suffix"]),
        )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        left, right = inputs["left"], inputs["right"]
        suffix = config["suffix"]
        left_keys, right_keys = _key_columns(config)
        right_names = _joined_right_names(left.columns, right.columns, right_keys, suffix)

        how = "inner" if config["how"] == "inner" else "left"

        if _a_key_is_typed_null(left, right, left_keys, right_keys):
            # A Null-typed key holds only nulls, so nothing pairs. It takes the other side's type so
            # `joined` keeps its usual columns and types.
            left_typed, right_typed = _null_keys_typed(left, right, left_keys, right_keys)
            joined = _pair(left_typed, right_typed, left_keys, right_keys, how, suffix)
            return Ports(
                {"joined": joined.select(*left.columns, *right_names.values()), "left_only": left, "right_only": right}
            )

        # maintain_order gives the documented row order. Both modes give the same columns.
        joined = _pair(left, right, left_keys, right_keys, how, suffix).select(*left.columns, *right_names.values())

        # Why an anti join: docs/design/tools.md#join
        left_only = _pair(left, right.select(right_keys), left_keys, right_keys, "anti", suffix)
        right_only = _pair(right, left.select(left_keys), right_keys, left_keys, "anti", suffix)

        return Ports({"joined": joined, "left_only": left_only, "right_only": right_only})


def _key_columns(config: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Each side's key columns, whichever of the two forms the configuration used."""
    shared = config["keys"]
    if shared is not None:
        return list(shared), list(shared)
    return list(config["left_keys"]), list(config["right_keys"])


def _refuse_keys_that_cannot_be_compared(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    shared: bool,
) -> None:
    """Refuse a key pair whose two types cannot be compared, naming both columns.

    An exact key against a float key is refused, naming the float column. Exact keys the tool pairs
    by value itself are accepted. Anything else is asked of polars directly with two empty frames,
    so it accepts exactly what a join accepts. A side typed Null is skipped: it takes the other
    side's type when the join runs.
    """
    field = "keys" if shared else "left_keys' and 'right_keys"
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        left_type, right_type = left.schema[left_key], right.schema[right_key]
        if pl.Null in (left_type, right_type):
            continue
        floating = _float_beside_exact(left_type, right_type)
        if floating is not None:
            side, column, exact_column = (
                ("left", left_key, right_key) if floating == "left" else ("right", right_key, left_key)
            )
            float_type = left_type if floating == "left" else right_type
            raise ConfigError(
                f"config field '{field}' pairs {side} column {column!r}, which holds {float_type}, with the "
                f"exact column {exact_column!r}, which holds {EXACT_TYPE}; a float is not an exact value, so "
                f"pairing them could match the wrong rows. Use cast_columns before this step to convert one "
                f"side, so both are {EXACT_TYPE} or both are Float64"
            )
        if _common_key_type(left_type, right_type) is not None:
            continue
        probe_left = pl.DataFrame(schema={"key": left_type})
        probe_right = pl.DataFrame(schema={"key": right_type})
        try:
            probe_left.join(probe_right, on="key", how="inner")
        except pl.exceptions.PolarsError:
            raise ConfigError(
                f"config field '{field}' pairs left column {left_key!r}, which holds {left_type}, with "
                f"right column {right_key!r}, which holds {right_type}; the two types cannot be "
                f"compared, so no row could pair. Use cast_columns before this step so both hold the "
                f"same type"
            ) from None


def _float_beside_exact(left_type: pl.DataType, right_type: pl.DataType) -> str | None:
    """Which side, 'left' or 'right', holds a float paired with an exact key on the other; None if neither."""
    if isinstance(right_type, pl.Decimal) and left_type.is_float():
        return "left"
    if isinstance(left_type, pl.Decimal) and right_type.is_float():
        return "right"
    return None


def _common_key_type(left_type: pl.DataType, right_type: pl.DataType) -> pl.DataType | None:
    """The type a key pair is compared in when the tool, not the engine, must make the two agree.

    Two exact keys of different places are compared at the larger places, and an exact key with a
    whole-number key at the exact side's places: SQL's = compares both by value. None for every
    other pair, which the engine compares as it is or refuses.
    """
    if left_type == right_type:
        return None
    left_exact, right_exact = isinstance(left_type, pl.Decimal), isinstance(right_type, pl.Decimal)
    if left_exact and right_exact:
        return exact_dtype(max(left_type.scale, right_type.scale))
    if left_exact and right_type.is_integer():
        return exact_dtype(left_type.scale)
    if right_exact and left_type.is_integer():
        return exact_dtype(right_type.scale)
    return None


def _pair(
    left: pl.DataFrame,
    right: pl.DataFrame,
    left_keys: Sequence[str],
    right_keys: Sequence[str],
    how: str,
    suffix: str,
) -> pl.DataFrame:
    """Join ``left`` to ``right`` on the key pairs, comparing exact keys by value.

    When no key pair needs it, this is the engine's own join. Otherwise both sides get a copy of
    each key in the type the pair is compared in, the join runs on the copies, and the copies are
    dropped, so every original column keeps its own type and digits. A value too wide for the
    larger places becomes null in its copy and pairs nothing, which is right: no value on the
    other side can equal it.
    """
    order = "left" if how == "anti" else "left_right"
    common = [
        _common_key_type(left.schema[left_key], right.schema[right_key])
        for left_key, right_key in zip(left_keys, right_keys, strict=True)
    ]
    if all(dtype is None for dtype in common):
        return left.join(
            right,
            left_on=list(left_keys),
            right_on=list(right_keys),
            how=how,
            suffix=suffix,
            nulls_equal=False,
            maintain_order=order,
        )
    taken = set(left.columns) | set(right.columns)
    copies = [_unused_name(f"__key_{index}", taken) for index in range(len(common))]

    def copied(frame: pl.DataFrame, keys: Sequence[str]) -> pl.DataFrame:
        return frame.with_columns(
            (pl.col(key) if dtype is None else pl.col(key).cast(dtype, strict=False)).alias(copy)
            for key, dtype, copy in zip(keys, common, copies, strict=True)
        )

    # The right keys repeat the left ones, so they are dropped before the join, as the engine does.
    right_side = right if how == "anti" else right.drop(right_keys)
    paired = copied(left, left_keys).join(
        copied(right, right_keys).select(*right_side.columns, *copies),
        on=copies,
        how=how,
        suffix=suffix,
        nulls_equal=False,
        maintain_order=order,
    )
    return paired.drop(copies)


def _unused_name(name: str, taken: set[str]) -> str:
    """``name``, lengthened with underscores until no column carries it."""
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _a_key_is_typed_null(
    left: pl.DataFrame, right: pl.DataFrame, left_keys: Sequence[str], right_keys: Sequence[str]
) -> bool:
    """Whether one side of a key pair is typed Null while the other side is not."""
    return any(
        (left.schema[left_key] == pl.Null) != (right.schema[right_key] == pl.Null)
        for left_key, right_key in zip(left_keys, right_keys, strict=True)
    )


def _null_keys_typed(
    left: pl.DataFrame, right: pl.DataFrame, left_keys: Sequence[str], right_keys: Sequence[str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Both frames, with each key column typed Null cast to its partner's type."""
    left_casts, right_casts = {}, {}
    for left_key, right_key in zip(left_keys, right_keys, strict=True):
        left_type, right_type = left.schema[left_key], right.schema[right_key]
        if left_type == pl.Null and right_type != pl.Null:
            left_casts[left_key] = right_type
        elif right_type == pl.Null and left_type != pl.Null:
            right_casts[right_key] = left_type
    return left.cast(left_casts), right.cast(right_casts)


def _joined_right_names(
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    right_keys: Sequence[str],
    suffix: str,
) -> dict[str, str]:
    """Each right-hand column that goes to `joined`, with the name it gets there.

    The right key columns are left out: after the join a key pair is one column, under the left name.
    """
    return {
        name: (f"{name}{suffix}" if name in left_columns else name) for name in right_columns if name not in right_keys
    }


def _refuse_unsettled_collision(left_columns: Sequence[str], right_names: Mapping[str, str]) -> None:
    """Refuse a right-hand column whose suffixed name is also taken."""
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
            "joined": pl.DataFrame(
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
        scenario="match each entry to its group, where one group is on the right twice",
        configuration={"keys": ["group_id"]},
        explanation="G1 is on the right twice, so each G1 entry comes out twice: four rows from three. "
        "The amount is repeated on each copy, not split. G4 has no partner, so it goes to 'left_only'. "
        "G3 has no partner on the left, so it goes to 'right_only'.",
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
            "joined": pl.DataFrame(
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
        scenario="match two tables whose key column has a different name on each side",
        configuration={"left_keys": ["period_start"], "right_keys": ["period"]},
        explanation="The key keeps its left name, 'period_start', on 'joined'; the right's 'period' "
        "is dropped there. Both sides have an 'amount' column, so the right one becomes "
        "'amount_right'. The unpaired ports keep each side's own columns, 'period' included.",
        inputs=inputs,
        outputs=outputs,
        check_row_order=True,
    )


REGISTRY.register(Join())
