"""A checked tree in, one polars column expression out.

This is the tree walk: operators, comparisons, ``in``, null tests, column references, literals and
function calls. Every type check is made here, against the schema when the caller has one. Every
refusal names the part of the expression at fault.
"""

import ast
import operator
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any, Final, NamedTuple

import polars as pl

from ..renames import FUNCTION_RENAMES
from .arguments import POSITIONS, Param, Value
from .errors import Refusal, error
from .families import (
    WILDCARD_FAMILIES,
    Family,
    Typed,
    arithmetic_dtype,
    combined_dtype,
    compatible,
    dtype_of_value,
    family_of_dtype,
    family_of_value,
    holding_phrase,
)
from .functions import ALLOWED_FUNCTIONS, FUNCTIONS, Function
from .parse import front_door, node_refusal
from .scan import COLUMN_FN, LITERAL_FN, Scanned
from .spelling import KEYWORD_LITERALS, MISSING, OPERATOR_SYMBOLS, Spelling, render_value


_BINARY_OPERATORS: Final[dict[type[ast.AST], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_COMPARISONS: Final[dict[type[ast.AST], Callable[[Any, Any], Any]]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

# Bounds that keep polars' own errors from escaping: a whole number must fit Int64, and a place or
# group count must be small. Measured on the value as written, sign included.
_WHOLE_NUMBER_MIN: Final[int] = -(2**63)
_WHOLE_NUMBER_MAX: Final[int] = 2**63 - 1


class Division(NamedTuple):
    """One '/' in an expression: its divisor as the author wrote it, and the divisor's column."""

    divisor: str
    expr: pl.Expr


class RowCheck(NamedTuple):
    """One function call's rule for the rows it has no answer for, ready to run on a frame.

    `failing` is true on such a row; `shown` gives the values a refusal quotes back. `problem` and
    `remedy` are the function's wording, filled in by :meth:`refusal` with `written`, the arguments as
    the author wrote them.
    """

    failing: pl.Expr
    shown: pl.Expr
    problem: str
    remedy: str
    written: tuple[str, ...]

    def refusal(self, found: Sequence[Any]) -> tuple[str, str]:
        """What is wrong, quoting back the values found, and what to do."""
        listed = ", ".join(render_value(value) for value in found)

        def fill(match: re.Match[str]) -> str:
            return listed if match[1] == "found" else self.written[int(match[1])]

        return _PLACEHOLDER.sub(fill, self.problem), _PLACEHOLDER.sub(fill, self.remedy)


# {0}, {1} ... and {found} in a function's wording; filled in one pass, so an argument that happens to
# be spelled like a placeholder stays as written.
_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{(\d+|found)\}")


def build(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None) -> tuple[pl.Expr, frozenset[str]]:
    """The front door, then the column expression for what survived it."""
    tree, scanned, schema = front_door(text, columns)
    expression = _Compiler(text, scanned, schema).compile(tree.body)
    return expression, frozenset(scanned.columns)


def divisions(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None) -> tuple[Division, ...]:
    """Every '/' in ``text``, in the order written, each with its divisor's own column expression."""
    tree, scanned, schema = front_door(text, columns)
    compiler = _Compiler(text, scanned, schema)
    compiler.compile(tree.body)
    return tuple(compiler.divisions)


def row_checks(text: str, columns: Mapping[str, pl.DataType] | Collection[str] | None) -> tuple[RowCheck, ...]:
    """Every function call in ``text`` that has rows with no answer, in the order written."""
    tree, scanned, schema = front_door(text, columns)
    compiler = _Compiler(text, scanned, schema)
    compiler.compile(tree.body)
    return tuple(compiler.row_checks)


class _Compiler:
    """Translates a checked tree into a polars expression. Every refusal about the language is made here.

    Recursive, which is safe only because the depth limit has already bounded the tree; only ever built
    on a tree `front_door` has returned.
    """

    __slots__ = ("_schema", "_spelling", "_text", "divisions", "row_checks")

    def __init__(
        self,
        text: str,
        scanned: Scanned,
        schema: dict[str, pl.DataType] | None,
    ) -> None:
        self._text = text
        self._spelling = Spelling(text, scanned)
        self._schema = schema
        #: Every '/' compiled, in the order written, so a tool can stop on a divisor of zero.
        self.divisions: list[Division] = []
        #: Every function call with rows it has no answer for, so a tool can stop on such a row.
        self.row_checks: list[RowCheck] = []

    def compile(self, node: ast.expr) -> Any:
        """What one node builds to, with its type discarded once it has been checked."""
        return self._typed(node).expr

    def _typed(self, node: ast.expr) -> Typed:
        """The expression for one node and the family it belongs to, dispatching on its type."""
        if isinstance(node, ast.BoolOp):
            return self._bool_op(node)
        if isinstance(node, ast.UnaryOp):
            return self._unary_op(node)
        if isinstance(node, ast.BinOp):
            return self._binary_op(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.Call):
            return self._call(node)
        if isinstance(node, ast.Name):
            return self._name(node)
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Tuple):
            raise error(
                self._text,
                "a list of values in parentheses is only meaningful on the right of 'in', as in [code] in ('A', 'B')",
            )
        raise node_refusal(self._text, node)

    # -- shapes ------------------------------------------------------------

    def _bool_op(self, node: ast.BoolOp) -> Typed:
        """`and` / `or`, left to polars so its three-valued logic applies.

        Each operand must be true/false; on numbers polars would do bitwise arithmetic, so `[qty] and [flag]`
        is refused and the author writes `[qty] > 0`.
        """
        spelling = OPERATOR_SYMBOLS[type(node.op)]
        operands: list[Typed] = []
        for value in node.values:
            part = self._typed(value)
            self._require(part, Family.BOOLEAN, f"'{spelling}' needs a truth value on both sides")
            operands.append(part)
        combine = operator.and_ if isinstance(node.op, ast.And) else operator.or_
        result = operands[0].expr.cast(pl.Boolean)
        for operand in operands[1:]:
            result = combine(result, operand.expr.cast(pl.Boolean))
        return Typed(result, Family.BOOLEAN, node, "gives a truth value")

    def _unary_op(self, node: ast.UnaryOp) -> Typed:
        if isinstance(node.op, ast.Not):
            operand = self._typed(node.operand)
            self._require(operand, Family.BOOLEAN, "'not' needs a truth value")
            return Typed(~operand.expr.cast(pl.Boolean), Family.BOOLEAN, node, "gives a truth value")

        # A signed number is one literal, so the most negative Int64 is spellable.
        folded = self._spelling.literal(node)
        if isinstance(folded, (int, float)) and not isinstance(folded, bool):
            self._check_whole_number(folded)
            return self._typed_value(folded, node)

        operand = _signed(self._typed(node.operand))
        self._require(operand, Family.NUMBER, "a minus sign needs a number")
        # A null of no type at all has no negation in this engine, which raises rather than
        # answering. So the null is handed back as it stands. There is nothing in it to negate, and
        # the answer is the same null either way.
        negative = operand.expr if operand.dtype == pl.Null else -operand.expr
        if operand.family is Family.NULL:
            # A minus sign in front of a null has nothing to negate, so the answer is still null.
            # It stays in the family that meets anything, the way the null it was written from
            # does, and every check below reads it as the null it is.
            return Typed(negative, Family.NULL, node, operand.holds, operand.dtype)
        # The type comes along, because a minus sign does not change it: the negative of a whole
        # number is a whole number, and the negative of a decimal is a decimal.
        return Typed(negative, Family.NUMBER, node, "gives a number", operand.dtype)

    def _binary_op(self, node: ast.BinOp) -> Typed:
        operation = type(node.op)
        if operation not in _BINARY_OPERATORS:  # pragma: no cover - the whitelist refused the rest
            raise node_refusal(self._text, node.op)
        symbol = OPERATOR_SYMBOLS[operation]
        left = self._typed(node.left)
        right = self._typed(node.right)
        for operand in (left, right):
            self._require_arithmetic(operand, symbol)
            if operation is ast.Div and isinstance(operand.dtype, pl.Decimal):
                raise error(
                    self._text,
                    f"{self._describe(operand)}, and '/' on a decimal column keeps the column's own number of "
                    f"places (100.00 / 3 gives 33.33, and 7 / 2 at no places gives 4) instead of dividing "
                    f"exactly; cast the column to Float64 first, with the cast_columns tool",
                )
        left, right = _signed(left), _signed(right)
        if operation is ast.Div:
            self._record_division(node.right, right)
        built = _BINARY_OPERATORS[operation](left.expr, right.expr)
        dtype = arithmetic_dtype(operation, left.dtype, right.dtype)
        return Typed(built, Family.NUMBER, node, "gives a number", dtype)

    def _compare(self, node: ast.Compare) -> Typed:
        if len(node.ops) != 1:
            raise error(
                self._text,
                "a chained comparison like 1 < [qty] < 3 is not part of this language: write the two comparisons "
                "joined with 'and', as in 1 < [qty] and [qty] < 3",
            )
        operation = node.ops[0]
        right_node = node.comparators[0]

        if isinstance(operation, (ast.Is, ast.IsNot)):
            return self._null_test(node, right_node, negated=isinstance(operation, ast.IsNot))
        if isinstance(operation, (ast.In, ast.NotIn)):
            return self._membership(node, right_node, negated=isinstance(operation, ast.NotIn))

        if type(operation) not in _COMPARISONS:  # pragma: no cover - the whitelist refused the rest
            raise node_refusal(self._text, operation)
        for side in (node.left, right_node):
            if isinstance(side, ast.Name) and side.id == "null":
                negated = isinstance(operation, ast.NotEq)
                raise error(
                    self._text,
                    f"comparing with null with '{OPERATOR_SYMBOLS[type(operation)]}' is never true: it gives null on "
                    f"every row, so a filter keeps nothing. To find a missing value write "
                    f"'is {'not ' if negated else ''}null', as in [code] is {'not ' if negated else ''}null",
                )
        left = self._typed(node.left)
        right = self._typed(right_node)
        if not compatible(left.family, right.family):
            raise error(
                self._text,
                f"a comparison needs both sides in the same type family, but {self._describe(left)} and "
                f"{self._describe(right)}",
            )
        self._refuse_timestamp_against_date(left, right, operation)
        built = _COMPARISONS[type(operation)](left.expr, right.expr)
        return Typed(built, Family.BOOLEAN, node, "gives a truth value")

    def _refuse_timestamp_against_date(self, left: Typed, right: Typed, operation: ast.cmpop | None) -> None:
        """Refuse a timestamp compared with a date where reading the date as midnight gives a wrong answer.

        Right for `>=` and `<`; wrong for `=`, `in`, `<=` (misses the day itself), `>` and `!=`.
        `operation` is None for `in`.
        """
        if isinstance(left.dtype, pl.Datetime) and right.dtype == pl.Date:
            timestamp, calendar_date, safe = left, right, (ast.GtE, ast.Lt)
        elif isinstance(right.dtype, pl.Datetime) and left.dtype == pl.Date:
            timestamp, calendar_date, safe = right, left, (ast.LtE, ast.Gt)
        else:
            return
        if operation is not None and isinstance(operation, safe):
            return
        raise error(
            self._text,
            f"{self._describe(timestamp)}, a timestamp, and {self._describe(calendar_date)}: a date compared with "
            f"a timestamp is read as midnight at its start, so a time later that day is not equal to it and "
            f"falls outside '<='. Cast the timestamp to Date first with the cast_columns tool, or write the "
            f"range as [ts] >= date'2025-01-01' and [ts] < date'2025-02-01'",
        )

    def _null_test(self, node: ast.Compare, right_node: ast.expr, *, negated: bool) -> Typed:
        """`is null` / `is not null`: the only tests never null themselves, so the way to find a missing value."""
        if not (isinstance(right_node, ast.Name) and right_node.id == "null"):
            raise error(
                self._text,
                "'is' may only be written as 'is null' or 'is not null': to compare values, use '=' or '!='",
            )
        left = self._typed(node.left)
        built = left.expr.is_not_null() if negated else left.expr.is_null()
        return Typed(built, Family.BOOLEAN, node, "gives a truth value")

    def _membership(self, node: ast.Compare, right_node: ast.expr, *, negated: bool) -> Typed:
        """`in` / `not in` over a bracketed list of literals.

        Only literals: a column on the right would be a hidden row-by-row join. A list mixing families is
        refused; a list mixing whole numbers and decimals is widened here, since polars types it from its
        first value.
        """
        candidates = right_node.elts if isinstance(right_node, ast.Tuple) else [right_node]
        values: list[Any] = []
        parts: list[Typed] = []
        for candidate in candidates:
            value = self._spelling.literal(candidate)
            if value is MISSING:
                raise error(
                    self._text,
                    "'in' needs a list of fixed values in parentheses, as in [code] in ('A', 'B')",
                )
            if isinstance(value, int) and not isinstance(value, bool):
                self._check_whole_number(value)
            values.append(value)
            parts.append(self._typed_value(value, candidate))

        listed = self._one_family(parts, "a list of values to match needs every value in the same type family")
        left = self._typed(node.left)
        if not compatible(left.family, listed):
            raise error(
                self._text,
                f"a membership test needs the listed values in the same type family as what they are matched "
                f"against, but {self._describe(left)} and the list is {listed.value}",
            )
        for part in parts:
            self._refuse_timestamp_against_date(left, part, None)
        if listed is Family.NUMBER and any(isinstance(value, float) for value in values):
            values = [value if value is None else float(value) for value in values]
        built = left.expr.is_in(list(values))
        if negated:
            built = ~built
        return Typed(built, Family.BOOLEAN, node, "gives a truth value")

    # -- leaves ------------------------------------------------------------

    def _name(self, node: ast.Name) -> Typed:
        if node.id in KEYWORD_LITERALS:
            return self._typed_value(KEYWORD_LITERALS[node.id], node)
        if node.id.lower() in KEYWORD_LITERALS:
            raise error(self._text, f"{node.id!r} is written {node.id.lower()!r} here, in lower case")
        raise error(
            self._text,
            f"{node.id!r} is not something this language knows: write a column as [{node.id}], text as 'value', or "
            f"one of true, false and null",
        )

    def _constant(self, node: ast.Constant) -> Typed:
        value = node.value
        if value is None or isinstance(value, bool):
            raise error(
                self._text,
                f"the Python literal {value!r} is not part of this language: write true, false or null in lower case",
            )
        if isinstance(value, int):
            self._check_whole_number(value)
        if isinstance(value, (int, float)):
            return self._typed_value(value, node)
        raise error(
            self._text,
            f"{value!r} is not a literal this language supports (it is a {type(value).__name__}): the literals are "
            f"whole numbers, decimals, single-quoted text, true, false, null and date'2026-01-01'",
        )

    def _column(self, node: ast.Call) -> Typed:
        name = self._spelling.column_name(node)
        if self._schema is None:
            return Typed(pl.col(name), Family.UNKNOWN, node, "holds an unknown type")
        dtype = self._schema[name]
        family = family_of_dtype(dtype)
        # A Categorical or Enum column is cast to String here, at the leaf, so every operation on it
        # meets a plain string. Why: docs/design/expressions.md#types-and-widths
        expression = pl.col(name)
        if family is Family.TEXT and dtype is not None and dtype != pl.String:
            expression = expression.cast(pl.String)
        return Typed(expression, family, node, f"holds {dtype}", dtype)

    def _typed_value(self, value: Any, node: ast.expr) -> Typed:
        """One literal, whose family is knowable from the text with or without a schema."""
        return Typed(
            pl.lit(value),
            family_of_value(value),
            node,
            holding_phrase(value),
            dtype_of_value(value),
        )

    # -- calls -------------------------------------------------------------

    def _call(self, node: ast.Call) -> Typed:
        if not isinstance(node.func, ast.Name):  # pragma: no cover - the whitelist refuses any other callee
            raise node_refusal(self._text, node.func)
        name = node.func.id

        if name == COLUMN_FN:
            return self._column(node)
        if name == LITERAL_FN:
            return self._typed_value(self._spelling.interned(node), node)
        function = FUNCTIONS.get(name)
        if function is None:
            available = ", ".join(sorted(ALLOWED_FUNCTIONS))
            renamed = FUNCTION_RENAMES.get(name)
            was = f"; it was renamed to {renamed!r}" if renamed is not None else ""
            raise error(self._text, f"{name!r} is not a function this language has{was}: the functions are {available}")

        arguments = node.args
        if function.variadic:
            if len(arguments) < 2:
                raise error(
                    self._text,
                    f"{name}() takes at least 2 arguments, but was given {len(arguments)}",
                )
        elif len(arguments) != len(function.params):
            arity = len(function.params)
            raise error(
                self._text,
                f"{name}() takes exactly {arity} argument{'s' if arity > 1 else ''}, but was given {len(arguments)}",
            )
        try:
            return self._apply(function, node)
        except Refusal as refusal:
            raise error(self._text, refusal.problem) from None

    def _apply(self, function: Function, node: ast.Call) -> Typed:
        """One whitelisted function: every argument read and checked, then the column built.

        A function with a fixed number of arguments reads them in order, and checks each one before
        reading the next, so the first argument at fault is the one reported. A variadic function
        compiles every argument first and then checks them, in order.
        """
        name = function.name
        if function.variadic:
            (param,) = function.params
            params: list[Param] = [param] * len(node.args)
            arguments = [self._typed(argument) for argument in node.args]
            if param.family is not None:
                for argument in arguments:
                    self._require(argument, param.family, f"{name}() {param.requirement}")
        else:
            params = list(function.params)
            arguments = [
                self._argument(name, param, argument, position)
                for position, (param, argument) in enumerate(zip(params, node.args, strict=True))
            ]

        operands = [argument for argument in arguments if isinstance(argument, Typed)]
        if function.result == "common":
            family = self._one_family(operands, f"{name}() {params[0].requirement}")
        else:
            family = function.result
        typed = {"dtype": operands[0].dtype} if function.typed else {}
        given = [self._built(param, argument) for param, argument in zip(params, arguments)]
        built = function.to_polars(*given, **typed)
        if function.no_answer is not None:
            self._record_row_check(function, node, given)

        if function.result_dtype == "common":
            dtype = combined_dtype([operand.dtype for operand in operands])
        elif function.result_dtype == "subject":
            dtype = operands[0].dtype
        else:
            dtype = function.result_dtype
        holds = f"gives {family.value}" if family not in WILDCARD_FAMILIES else "holds an unknown type"
        return Typed(built, family, node, holds, dtype)

    def _argument(self, name: str, param: Param, node: ast.expr, position: int) -> Any:
        """One argument of a fixed-arity function: a checked operand, or a value read from the file."""
        if isinstance(param, Value):
            typed = self._typed(node)
            if param.family is not None:
                self._require(typed, param.family, f"{name}() {param.requirement}")
            if param.whole is not None:
                self._require_whole(typed, f"{name}() {param.whole}")
            if param.no_time_zone and isinstance(typed.dtype, pl.Datetime) and typed.dtype.time_zone is not None:
                raise error(
                    self._text,
                    f"{self._describe(typed)}, a timestamp in time zone {typed.dtype.time_zone!r}, but {name}() works "
                    f"on dates, and which day such a timestamp falls on depends on the zone. Cast it to Date "
                    f"first with the cast_columns tool and its in_zone setting, so the day is the one you choose",
                )
            return typed
        return param.read(name, self._spelling.literal(node), POSITIONS[position])

    def _built(self, param: Param, argument: Any) -> Any:
        """What a function's builder is given for one argument; text is cast to a plain String.

        Needed when no schema was given, so a compact text column cannot reach a string function uncast.
        """
        if not isinstance(param, Value):
            return argument
        if param.family is Family.TEXT:
            return argument.expr.cast(pl.String)
        return argument.expr

    # -- type requirements -------------------------------------------------

    def _require(self, typed: Typed, family: Family, requirement: str) -> None:
        """Refuse an operand outside the family an operation needs: which operand, what it holds, what was needed."""
        if typed.family in WILDCARD_FAMILIES or typed.family is family:
            return
        raise error(self._text, f"{self._describe(typed)}, but {requirement}")

    def _require_whole(self, typed: Typed, requirement: str) -> None:
        """Refuse a count of months that cannot be shown to be whole.

        Half a month is not a distance; a 1.5 usually means a forgotten division. Refused: a decimal column or
        literal, a decimal from arithmetic, abs(), round() or coalesce(), any division, and a count of unknown
        type. A null gives a null date. Nothing is refused without a schema.
        """
        if typed.family in WILDCARD_FAMILIES or self._schema is None:
            return
        if typed.dtype is None:
            raise error(
                self._text,
                f"{self._describe(typed)}, and nothing here can work out whether that is whole, but {requirement}",
            )
        if typed.dtype.is_integer() or typed.dtype == pl.Null:
            return
        raise error(self._text, f"{self._describe(typed)}, but {requirement}")

    def _require_arithmetic(self, typed: Typed, symbol: str) -> None:
        """Arithmetic needs numbers. `+` on text is refused: polars would join '100' and '20' into '10020'."""
        if typed.family in WILDCARD_FAMILIES or typed.family is Family.NUMBER:
            return
        hint = (
            ", and to join two pieces of text use concat() rather than an arithmetic operator"
            if typed.family is Family.TEXT
            else ""
        )
        raise error(
            self._text,
            f"{self._describe(typed)}, but arithmetic with '{symbol}' needs numbers on both sides{hint}",
        )

    def _one_family(self, parts: Sequence[Typed], requirement: str) -> Family:
        """The one family a list of operands agrees on, refusing the first that disagrees.

        polars would turn a number into text to sit beside text, without a word.
        """
        settled: Typed | None = None
        for part in parts:
            if part.family in WILDCARD_FAMILIES:
                continue
            if settled is None:
                settled = part
            elif part.family is not settled.family:
                raise error(
                    self._text,
                    f"{requirement}, but {self._describe(settled)} and {self._describe(part)}",
                )
        if settled is not None:
            return settled.family
        return Family.NULL if all(part.family is Family.NULL for part in parts) else Family.UNKNOWN

    def _describe(self, typed: Typed) -> str:
        """One operand named the way its author wrote it, with what it holds."""
        return f"{self._spelling.render(typed.node)} {typed.holds}"

    # -- helpers -----------------------------------------------------------

    def _record_division(self, node: ast.expr, divisor: Typed) -> None:
        """Note a divisor for the tool to check row by row, or refuse one written as zero."""
        written = self._spelling.literal(node)
        if isinstance(written, (int, float)) and not isinstance(written, bool) and written == 0:
            raise error(self._text, f"'/' divides by {self._spelling.render(node)}, which is zero, so it has no answer")
        self.divisions.append(Division(self._spelling.render(node), divisor.expr))

    def _record_row_check(self, function: Function, node: ast.Call, given: list[Any]) -> None:
        """Note the rows a function call has no answer for, with its arguments spelled as written."""
        rule = function.no_answer
        assert rule is not None
        written = tuple(self._spelling.render(argument) for argument in node.args)
        self.row_checks.append(RowCheck(rule.when(*given), given[rule.shows], rule.problem, rule.remedy, written))

    def _check_whole_number(self, value: int | float) -> None:
        """Refuse a whole number no column could hold, measured on the value as written."""
        if isinstance(value, int) and not _WHOLE_NUMBER_MIN <= value <= _WHOLE_NUMBER_MAX:
            raise error(
                self._text,
                f"the whole number {value} is outside the range this language holds, which is {_WHOLE_NUMBER_MIN} to "
                f"{_WHOLE_NUMBER_MAX}",
            )


def _signed(typed: Typed) -> Typed:
    """An unsigned whole number cast to a signed 64-bit one before arithmetic, so it cannot wrap.

    len() and some columns hold unsigned numbers, and 3 - 5 on those wraps to 4,294,967,294 instead
    of -2. Any other value is handed back as it is.
    """
    if typed.dtype is None or not typed.dtype.is_unsigned_integer():
        return typed
    return typed._replace(expr=typed.expr.cast(pl.Int64), dtype=pl.Int64())
