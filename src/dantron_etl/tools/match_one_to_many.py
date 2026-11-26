"""MatchOneToMany: pair each whole with the one set of its candidate parts whose amounts add up to it.

Each row of `candidates` is one possible (whole, part) pair. For each whole, every set of 1 to
`max_parts` of its parts is tried. Exactly one set that adds up is a match; none, or more than one,
is set aside with a reason. Amounts are compared exactly, as whole numbers of their smallest unit.
"""

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
    require_columns,
    require_not_blank,
)

NO_SET: Final[str] = "no set adds up"
MANY_SETS: Final[str] = "more than one set adds up"
OUTSIDE: Final[str] = "not in the matched set"
CLAIMED: Final[str] = "a part is claimed by more than one whole"
MISSING_KEY: Final[str] = "missing key"
MISSING_AMOUNT: Final[str] = "missing amount"
TOO_MANY: Final[str] = "more candidates than the search allows"

# The status of a whole whose search found exactly one set. Never written to a row.
_ONE_SET: Final[str] = "one set"

# How many key values a data error quotes.
_KEYS_SHOWN: Final[int] = 5

_KEY_FIELDS: Final[tuple[str, ...]] = ("whole_key", "part_key")
_AMOUNT_FIELDS: Final[tuple[str, ...]] = ("whole_amount", "part_amount")


class MatchOneToMany:
    """Choose, for each whole, the single set of candidate parts that adds up to it."""

    name = "match_one_to_many"
    description = "Pair each whole with the one set of its candidate parts whose amounts add up to it."
    category = "preparation"
    input_ports = ("candidates",)
    output_ports = ("matched", "rejected")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "whole_key": ConfigField(
                type="string",
                description="The column naming the whole on each row, such as a deposit id.",
                required=True,
                non_empty=True,
            ),
            "part_key": ConfigField(
                type="string",
                description="The column naming the part on each row, such as a payment id.",
                required=True,
                non_empty=True,
            ),
            "whole_amount": ConfigField(
                type="string",
                description="The column holding the whole's amount. A whole number or an exact Decimal.",
                required=True,
                non_empty=True,
            ),
            "part_amount": ConfigField(
                type="string",
                description="The column holding the part's amount. A whole number or an exact Decimal.",
                required=True,
                non_empty=True,
            ),
            "max_parts": ConfigField(
                type="integer",
                description="The most parts one whole may be made of. At least 1.",
                required=True,
            ),
            "tolerance": ConfigField(
                type="number",
                description="How far the parts' total may be from the whole and still add up, in the "
                "amounts' own units. 0 or more.",
                default=0,
            ),
            "max_candidates": ConfigField(
                type="integer",
                description="A whole with more candidate parts than this is not searched, and its rows "
                "are rejected. At least 1.",
                default=25,
            ),
            "reason_column": ConfigField(
                type="string",
                description="The name of the column added to `rejected`, saying why each row is there.",
                required=True,
                non_empty=True,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use match_one_to_many to pair one whole with the several parts that make it up, such "
                "as one bank deposit with the payments it holds. Each input row is one possible "
                "(whole, part) pair, usually made by a join on a shared key and a filter on a date "
                "window. For each whole, every set of 1 to 'max_parts' of its parts is tried. When "
                "exactly one set adds up to the whole, its rows go to `matched`. Every other row goes "
                "to `rejected`, with the reason in 'reason_column'. Nothing is guessed: a whole with "
                "no answer, or more than one, is left for a person."
            ),
            examples=(_worked_example(),),
            notes=(
                "No SQL operation does this. Missing values follow SQL's '=': a missing key matches nothing.",
                "A set adds up when the gap between its total and the whole is at most 'tolerance'. "
                "Amounts are compared exactly; nothing is rounded.",
                "No set adds up: every row of the whole is rejected with 'no set adds up'.",
                "Two or more sets add up: every row of the whole is rejected with 'more than one set adds up'.",
                "Exactly one set adds up: its rows are matched, and the whole's other rows are rejected "
                "with 'not in the matched set'.",
                "A part in the matched sets of two or more wholes: every row of each of those wholes is "
                "rejected with 'a part is claimed by more than one whole'. Nothing is tried again.",
                "A row missing its whole key or its part key is rejected with 'missing key', even when "
                "its amount is missing too.",
                "A row missing its part amount, or whose whole has no amount, is rejected with 'missing amount'.",
                "Rows rejected for a missing key or amount take no part in any search, do not count "
                "towards 'max_candidates', and keep their reason whatever happens to their whole.",
                "A whole with more candidates than 'max_candidates' is not searched. Its rows are "
                "rejected with 'more candidates than the search allows'.",
                "These stop the step, naming the keys: the same whole and part on two rows; one whole "
                "with two different amounts; one part with two different amounts. A missing amount "
                "beside a value counts as different.",
                "Zero and negative amounts are allowed. A part of zero can join any set, so it makes a "
                "second set that adds up, if there is room for it under 'max_parts'.",
                "A set has at least one part, so a whole of zero is not matched by no parts.",
                "A Float64 amount is refused: totals of decimal-point numbers do not add up exactly. "
                "Convert it to Decimal with cast_columns first. A whole-number amount beside a Decimal "
                "one is compared by value.",
                "The answer does not depend on the order of the rows. Both ports keep the input order.",
            ),
            field_help={
                "whole_key": "For example 'deposit_id'. It must be a different column from 'part_key'.",
                "part_key": "For example 'payment_id'.",
                "whole_amount": "For example 'deposit_amount'. Every row of one whole must give the same amount.",
                "part_amount": "For example 'payment_amount'. Every row of one part must give the same amount.",
                "max_parts": "For example 3. A larger number finds bigger sets, and more ways for a "
                "whole to add up twice.",
                "tolerance": "For example 0.01 to allow a cent either way. Left out, it is 0: the "
                "total must equal the whole exactly.",
                "max_candidates": "Left out, it is 25. The search grows fast with the number of "
                "candidates; narrow them with a filter rather than raising this a lot.",
                "reason_column": "For example 'reason'. It must be a name the input does not have.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the same key column twice, blank names, and numbers out of range."""
        for field in (*_KEY_FIELDS, *_AMOUNT_FIELDS, "reason_column"):
            require_not_blank(config[field], field)
        if config["whole_key"] == config["part_key"]:
            raise ConfigError(
                f"config fields 'whole_key' and 'part_key' both name {config['whole_key']!r}; a pair needs "
                f"a whole key column and a different part key column"
            )
        for field in ("max_parts", "max_candidates"):
            if config[field] is None or config[field] < 1:
                raise ConfigError(
                    f"config field {field!r} is {config[field]!r}; it must be a whole number of 1 or more"
                )
        tolerance = config["tolerance"]
        if tolerance is None or not math.isfinite(tolerance) or tolerance < 0:
            raise ConfigError(
                f"config field 'tolerance' is {tolerance!r}; it must be a number of 0 or more, such as 0 or 0.01"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a column that is not there, an amount that is not exact, and a reason column in use."""
        frame = inputs["candidates"]
        for field in (*_KEY_FIELDS, *_AMOUNT_FIELDS):
            require_columns(frame, [config[field]], field, "candidates")
        for field in _AMOUNT_FIELDS:
            column = config[field]
            dtype = frame.schema[column]
            if dtype.is_integer() or isinstance(dtype, pl.Decimal):
                continue
            if dtype.is_float():
                raise ConfigError(
                    f"config field {field!r} names column {column!r}, which holds {dtype} numbers; totals "
                    f"of decimal-point numbers do not add up exactly. Convert it to Decimal with "
                    f"cast_columns before this step"
                )
            raise ConfigError(
                f"config field {field!r} names column {column!r}, which holds {dtype}; an amount must be a "
                f"whole number or an exact Decimal. Convert it with cast_columns before this step"
            )
        reason = config["reason_column"]
        if reason in frame.columns:
            raise ConfigError(
                f"config field 'reason_column' names column {reason!r}, which the input already has; "
                f"choose a name the input does not use"
            )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        """Refuse contradictory rows, set aside rows that cannot be searched, search each whole, then settle claims."""
        frame = inputs["candidates"]
        whole_dtype = frame.schema[config["whole_amount"]]
        part_dtype = frame.schema[config["part_amount"]]

        # Amounts as whole numbers of each column's own smallest unit.
        work = frame.select(
            pl.int_range(pl.len(), dtype=pl.get_index_type()).alias("i"),
            pl.col(config["whole_key"]).alias("w"),
            pl.col(config["part_key"]).alias("p"),
            _units_of(config["whole_amount"], whole_dtype).alias("wa"),
            _units_of(config["part_amount"], part_dtype).alias("pa"),
        )
        _refuse_contradictions(frame, work, config)

        # The smallest unit common to both amount columns and the tolerance.
        tolerance = (
            Decimal(repr(config["tolerance"]))
            if isinstance(config["tolerance"], float)
            else Decimal(config["tolerance"])
        )
        whole_places, part_places = _places(whole_dtype), _places(part_dtype)
        places = max(whole_places, part_places, _decimal_places(tolerance))
        whole_scale = 10 ** (places - whole_places)
        part_scale = 10 ** (places - part_places)
        tolerance_units = _decimal_units(tolerance, places)

        missing = (
            pl.when(pl.col("w").is_null() | pl.col("p").is_null())
            .then(pl.lit(MISSING_KEY))
            .when(pl.col("wa").is_null() | pl.col("pa").is_null())
            .then(pl.lit(MISSING_AMOUNT))
            .otherwise(pl.lit(None, dtype=pl.String))
        )
        work = work.with_columns(reason=missing)
        searchable = pl.col("reason").is_null().sum().over("w")
        work = work.with_columns(
            reason=pl.when(pl.col("reason").is_null() & (searchable > config["max_candidates"]))
            .then(pl.lit(TOO_MANY))
            .otherwise(pl.col("reason"))
        )
        settled = work.filter(pl.col("reason").is_not_null()).select("i", "reason")
        to_search = work.filter(pl.col("reason").is_null())

        groups = to_search.group_by("w", maintain_order=True).agg("i", "pa", pl.col("wa").first())
        statuses: list[str] = []
        chosen: list[int] = []
        for rows, amounts, whole in groups.select("i", "pa", "wa").iter_rows():
            found, mask = _search(
                [amount * part_scale for amount in amounts],
                whole * whole_scale,
                tolerance_units,
                config["max_parts"],
            )
            statuses.append((NO_SET, _ONE_SET, MANY_SETS)[found])
            if found == 1:
                chosen.extend(row for bit, row in enumerate(rows) if mask >> bit & 1)

        index_type = pl.get_index_type()
        per_row = (
            pl.DataFrame(
                {
                    "g": pl.Series(range(groups.height), dtype=pl.Int64),
                    "i": groups["i"],
                    "status": pl.Series(statuses, dtype=pl.String),
                }
            )
            .explode("i", empty_as_null=False)
            .join(to_search.select("i", "p"), on="i")
            .with_columns(chosen=pl.col("i").is_in(pl.Series(chosen, dtype=index_type).implode()))
        )
        tentative = per_row.filter(pl.col("chosen"))
        claimed = tentative.filter(pl.len().over("p") > 1)["g"].unique()
        searched = per_row.select(
            "i",
            reason=pl.when(pl.col("status") != _ONE_SET)
            .then(pl.col("status"))
            .when(pl.col("g").is_in(claimed.implode()))
            .then(pl.lit(CLAIMED))
            .when(pl.col("chosen"))
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.lit(OUTSIDE)),
        )

        reasons = pl.concat([settled, searched.cast({"i": index_type})]).sort("i")["reason"]
        rejected_mask = reasons.is_not_null()
        return Ports(
            {
                "matched": frame.filter(~rejected_mask),
                "rejected": frame.with_columns(reasons.alias(config["reason_column"])).filter(rejected_mask),
            }
        )


def _units_of(column: str, dtype: pl.DataType) -> pl.Expr:
    """An amount column as whole numbers of its own smallest unit: an exact column's stored digits."""
    if isinstance(dtype, pl.Decimal):
        return pl.col(column).to_physical().cast(pl.Int128)
    return pl.col(column).cast(pl.Int128)


def _places(dtype: pl.DataType) -> int:
    return (dtype.scale or 0) if isinstance(dtype, pl.Decimal) else 0


def _decimal_places(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    assert isinstance(exponent, int)
    return max(0, -exponent)


def _decimal_units(value: Decimal, places: int) -> int:
    """``value`` as a whole number of units of 10 ** -places, exactly. ``places`` holds all its digits."""
    sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    units = int("".join(str(digit) for digit in digits) or "0") * 10 ** (exponent + places)
    return -units if sign else units


def _search(amounts: Sequence[int], target: int, tolerance: int, max_parts: int) -> tuple[int, int]:
    """How many sets of 1 to ``max_parts`` amounts are within ``tolerance`` of ``target``: 0, 1 or 2 (meaning 2 or more).

    With exactly one, the second value is that set, as a bit mask over ``amounts``. Meet in the
    middle: every small enough set of each half is listed once, the right half's totals sorted by
    set size, and each left set looks up the right totals that complete it. It never relies on
    totals growing with size, so negative amounts are safe. It stops at the second set.
    """
    size = len(amounts)
    most = min(max_parts, size)
    half = size // 2
    left = _subsets(amounts[:half], 0, most)
    right_by_size: list[tuple[list[int], list[int]]] = []
    right = sorted(_subsets(amounts[half:], half, most))
    for count in range(most + 1):
        same = [(total, mask) for total, used, mask in right if used == count]
        right_by_size.append(([total for total, _ in same], [mask for _, mask in same]))

    found = 0
    chosen = 0
    for total, used, mask in left:
        low = target - tolerance - total
        high = target + tolerance - total
        for count in range(0 if used else 1, most - used + 1):
            totals, masks = right_by_size[count]
            start = bisect_left(totals, low)
            end = bisect_right(totals, high)
            if end > start:
                found += end - start
                if found >= 2:
                    return 2, 0
                chosen = mask | masks[start]
    return found, chosen


def _subsets(amounts: Sequence[int], offset: int, most: int) -> list[tuple[int, int, int]]:
    """Every set of at most ``most`` of ``amounts``, the empty set included, as (total, size, bit mask)."""
    sets = [(0, 0, 0)]
    for position, amount in enumerate(amounts):
        bit = 1 << (offset + position)
        sets += [(total + amount, used + 1, mask | bit) for total, used, mask in sets if used < most]
    return sets


def _refuse_contradictions(frame: pl.DataFrame, work: pl.DataFrame, config: Mapping[str, Any]) -> None:
    """Stop on the same pair twice, a whole with two amounts, or a part with two amounts."""
    whole_key, part_key = config["whole_key"], config["part_key"]
    keyed = work.filter(pl.col("w").is_not_null() & pl.col("p").is_not_null())
    repeated = keyed.filter(pl.len().over("w", "p") > 1)
    if repeated.height:
        raise DataError(
            f"the same whole and part are on more than one row (columns {whole_key!r} and {part_key!r}), "
            f"for whole(s) {_shown(repeated['w'])}; each pair may be a candidate once. Remove the repeated "
            f"rows first, with remove_duplicates",
            offending=frame[repeated["i"]],
            total=frame.height,
        )
    checks = (
        ("w", "wa", "whole", whole_key, config["whole_amount"]),
        ("p", "pa", "part", part_key, config["part_amount"]),
    )
    for key, amount, noun, key_column, amount_column in checks:
        named = work.filter(pl.col(key).is_not_null())
        differing = named.filter(pl.col(amount).n_unique().over(key) > 1)
        if differing.height:
            raise DataError(
                f"column {amount_column!r} gives one {noun} two or more different amounts (a missing "
                f"amount beside a value counts as different), for {noun}(s) {_shown(differing[key])} in "
                f"column {key_column!r}; every row of one {noun} must give the same amount",
                offending=frame[differing["i"]],
                total=frame.height,
            )


def _shown(keys: pl.Series) -> str:
    values = keys.unique(maintain_order=True)
    shown = ", ".join(str(value) for value in values.head(_KEYS_SHOWN).to_list())
    return shown if values.len() <= _KEYS_SHOWN else f"{shown} and {values.len() - _KEYS_SHOWN} more"


def _worked_example() -> WorkedExample:
    money = pl.Decimal(12, 2)
    schema = {
        "deposit_id": pl.String,
        "deposit_amount": money,
        "payment_id": pl.String,
        "payment_amount": money,
    }
    rows = [
        ("W1", "250.00", "A", "100.00"),
        ("W1", "250.00", "B", "150.00"),
        ("W1", "250.00", "E", "40.00"),
        ("W2", "300.00", "C", "200.00"),
        ("W2", "300.00", "D", "100.00"),
        ("W2", "300.00", "F", "150.00"),
        ("W2", "300.00", "G", "150.00"),
        ("W3", "90.00", "H", "45.00"),
        ("W3", "90.00", "I", "40.00"),
    ]

    def table(chosen: Sequence[tuple[str, str, str, str]]) -> pl.DataFrame:
        return pl.DataFrame(
            [(whole, Decimal(amount), part, Decimal(paid)) for whole, amount, part, paid in chosen],
            schema=schema,
            orient="row",
        )

    rejected = table(rows[2:]).with_columns(
        reason=pl.Series([OUTSIDE, MANY_SETS, MANY_SETS, MANY_SETS, MANY_SETS, NO_SET, NO_SET], dtype=pl.String)
    )
    return WorkedExample(
        scenario="match each deposit to the one set of payments that adds up to it",
        configuration={
            "whole_key": "deposit_id",
            "part_key": "payment_id",
            "whole_amount": "deposit_amount",
            "part_amount": "payment_amount",
            "max_parts": 3,
            "reason_column": "reason",
        },
        explanation="W1 is 250.00, and only A + B adds up to it, so A and B are matched and E is left "
        "over. W2 is 300.00, which is C + D and also F + G, so nothing is guessed and all four rows are "
        "set aside. No set of W3's payments adds up to 90.00.",
        inputs=Ports({"candidates": table(rows)}),
        outputs=Ports({"matched": table(rows[:2]), "rejected": rejected}),
        check_row_order=True,
    )


REGISTRY.register(MatchOneToMany())
