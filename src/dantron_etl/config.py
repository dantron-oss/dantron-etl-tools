"""A tool's configuration template, and the checks that hold a configuration to it.

A tool describes its settings as a mapping of ``ConfigField`` entries: a type, a description,
whether the setting is required, a default. This module checks a configuration against that
template, fills in the defaults, and finds the expressions a configuration holds. It needs no frame,
so everything here can run when a pipeline file is loaded.

The template is enforced by the boundary rather than by each tool, because a tool with no
cross-field rules has no reason to write anything of its own, and a rule that holds only when an
author remembers to apply it is a convention rather than a rule. A configuration field can declare
itself an *expression*, which is the seam to the small expression language this package also ships:
a caller that loads a pipeline definition can lift the column references out of a field like that
and check them against an upstream frame before a single row runs through it. A field can also
describe the shape *below* itself -- an array's element type, an object's members -- because the
common configuration shape is an ordered list of small objects, and a seam that reaches only the
top level of such a configuration reaches nothing at all.
"""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .errors import ConfigError
from .expr import ExpressionError, compile_expr

__all__ = [
    "FIELD_TYPES",
    "ConfigField",
    "apply_defaults",
    "expression_fields",
    "expression_values",
    "require_not_blank",
    "with_base_dir",
    "validate_against_template",
]


# One check per declared type. A boolean is refused by both numeric types because ``bool`` is an
# ``int`` subclass in Python, which is the same trap the expression language had to sidestep for its
# own whole-number arguments; the two must agree, or a template accepts ``true`` where it says it
# takes a count.
_TYPE_CHECKS: Final[dict[str, Any]] = {
    "string": lambda value: isinstance(value, str),
    "expression": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}

# The field types a template may declare. Kept as a published set, the same way expr.py publishes
# its own whitelists, so that a template author sees exactly what is available rather than
# discovering the boundary by trial and error. "expression" is one of these types rather than a
# separate flag on the field, so a field cannot claim to be an expression in one place and a plain
# string in another. Derived from the checks above rather than written out beside them, so the
# published set and the set that is actually enforced cannot drift apart -- a type in one and not
# the other would be a field that validates nothing while looking as if it does.
FIELD_TYPES: Final[frozenset[str]] = frozenset(_TYPE_CHECKS)


# The types ``non_empty`` can mean anything about. A count of zero or a number of zero are values in
# full, not an absence of one, so ``non_empty`` on either would constrain nothing -- the same silent
# no-op ``_check_spec`` already refuses for ``items`` outside an array and ``fields`` outside an
# object.
_SIZED_TYPES: Final[frozenset[str]] = frozenset({"string", "expression", "array", "object"})


@dataclass(frozen=True, slots=True)
class ConfigField:
    """One entry in a tool's configuration template.

    ``type`` is a member of :data:`FIELD_TYPES`, and ``"expression"`` is one of those types rather
    than a separate boolean, so :func:`expression_fields` has exactly one place to read instead of
    two that could disagree.

    ``items`` describes an array's element and ``fields`` describes an object's members, which is
    the smallest pair of additions that makes the common configuration shape -- an ordered list of
    small objects, as used for ordered rules, lists of aggregations, lists of key pairs -- say
    anything about itself. Without them such a configuration is one untyped ``array`` field, so
    nothing inside it is type checked, required checked, or compiled, and the expressions it holds
    are invisible to a caller pre-flighting them.

    A default is not filled inside ``items`` or ``fields``: an element that does not exist has
    nothing to fill, and inventing one would put a rule in an ordered list that nobody wrote.

    ``required`` and ``non_empty`` answer two different questions: whether a value arrived at all,
    and whether the value that arrived carries anything. ``required=True`` accepts ``[]``, ``""``
    and ``{}`` -- each is a value -- and ``non_empty=True`` is how a field refuses them, on its own
    or alongside ``required``, without teaching ``required`` a second meaning.
    """

    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Sequence[str] | None = None
    items: "ConfigField | None" = None
    fields: "Mapping[str, ConfigField] | None" = None
    non_empty: bool = False
    #: The field holds the path of a file the tool reads or writes. A run given ``base_dir`` reads a
    #: relative path in such a field from that folder instead of the working directory.
    file_path: bool = False


def expression_fields(template: Mapping[str, ConfigField]) -> tuple[str, ...]:
    """Paths of the fields in ``template`` that hold an expression, in template order.

    The declared field type is the whole seam to the expression language, and the path is how a
    field nested inside an array or an object is named: ``"rules[].when"`` is the ``when`` member
    of an element of the ``rules`` array. A path carries no index, because a template describes
    what *may* be there and no particular element exists in it -- which is exactly the question a
    form renderer or a documentation page asks. For what a configuration *does* hold, and where,
    use :func:`expression_values`.
    """
    return tuple(_expression_paths(template, prefix=""))


def _expression_paths(template: Mapping[str, ConfigField], prefix: str) -> Iterator[str]:
    for name, spec in template.items():
        yield from _spec_expression_paths(_joined(prefix, name), spec)


def _spec_expression_paths(path: str, spec: ConfigField) -> Iterator[str]:
    if spec.type == "expression":
        yield path
    elif spec.type == "array" and spec.items is not None:
        yield from _spec_expression_paths(f"{path}[]", spec.items)
    elif spec.type == "object" and spec.fields is not None:
        yield from _expression_paths(spec.fields, prefix=path)


def expression_values(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """Every expression ``config`` actually holds, as ``(path, text)`` pairs in template order.

    Separate from :func:`expression_fields` because the two answer different questions and only one
    of them can be answered by a template. A concrete index -- ``rules[0].when`` -- is the only
    form of path a caller can put in front of a person who has to go and fix the expression, and an
    index cannot exist in a template, so the reportable path needs the configuration.

    This is what makes the seam reach a configuration that has no top-level expression field at
    all: a caller lifts the references out of every pair, checks them against the upstream frame,
    and reports the ones that miss by the path they sit at, before a single row is read.
    """
    return tuple(_expression_values(template, config, prefix=""))


def _expression_values(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    prefix: str,
) -> Iterator[tuple[str, str]]:
    for name, spec in template.items():
        if name in config:
            yield from _spec_expression_values(_joined(prefix, name), spec, config[name])


def _spec_expression_values(path: str, spec: ConfigField, value: Any) -> Iterator[tuple[str, str]]:
    if value is None:
        return
    if spec.type == "expression" and isinstance(value, str):
        yield path, value
    elif spec.type == "array" and spec.items is not None and isinstance(value, list):
        for index, item in enumerate(value):
            yield from _spec_expression_values(f"{path}[{index}]", spec.items, item)
    elif spec.type == "object" and spec.fields is not None and isinstance(value, dict):
        yield from _expression_values(spec.fields, value, prefix=path)


def apply_defaults(template: Mapping[str, ConfigField], config: Mapping[str, Any]) -> dict[str, Any]:
    """``config`` with every field of ``template`` present -- its default, or ``None``.

    Every field, not only the ones carrying a declared default, because the alternative makes
    ``execute`` read its own configuration two ways: subscript the field with a default, ``.get``
    the field beside it without one, a distinction with no meaning that every author has to
    rediscover field by field. Filling all of them makes "was this configured?" one expression,
    ``config["x"] is None``, everywhere.

    That only composes because :func:`validate_against_template` accepts ``None`` for an optional
    field: a filled configuration is handed to a tool's own ``validate_config``, which validates it
    against the same template a second time, and it has to survive that.

    Nothing is filled inside an array or an object. There is no element to fill a member of, and
    inventing one would add an entry to an ordered list that nobody wrote.

    Public because a loader assembling a configuration from a file wants the same filling the
    boundary does.
    """
    filled = dict(config)
    for name, spec in template.items():
        if name not in filled:
            filled[name] = spec.default
    return filled


def _joined(prefix: str, name: str) -> str:
    """A field's reportable path: ``name`` at the top level, ``rules[0].when`` inside a value."""
    return f"{prefix}.{name}" if prefix else name


def require_not_blank(value: str, field: str) -> None:
    """Refuse a name that is empty or nothing but whitespace, naming the field that holds it."""
    if not value.strip():
        raise ConfigError(f"config field '{field}' must not be blank")


def validate_against_template(template: Mapping[str, ConfigField], config: Mapping[str, Any]) -> None:
    """Everything a template can express about ``config``, checked uniformly for every field.

    An unknown field, a missing required one, a value of the wrong type, a value outside its
    ``enum``, and an expression that will not compile are all refused here -- the last one compiled
    with no known columns, so bad *syntax* is caught at this boundary while a missing *column* stays
    the caller's problem, because only the caller knows the frame the tool is about to run against.

    A string still holding ``${...}`` anywhere inside it -- including nested inside an array or an
    object value -- is refused too, regardless of the field's declared type. Such a placeholder is
    meant to be resolved by whoever assembles the configuration; it is not this module's job to
    resolve it, but it is exactly this module's job to refuse one that arrives unresolved, because a
    placeholder like that would otherwise compile as an ordinary piece of text, match nothing, and
    leave the run looking as if it had succeeded.

    Wherever a field declares the shape below itself, a nested value is held to exactly these same
    rules and the refusal names it by its path -- ``rules[0].when`` -- because a configuration whose
    expressions all sit inside an array of objects is the common case, not the exotic one.

    A field whose declared type is not in :data:`FIELD_TYPES` is refused rather than skipped. The
    lookup used to fail open, which meant a template author reaching for the obvious name of a type
    that does not exist got a field accepting anything at all.

    ``None`` is accepted for an optional field and still refused for a required one. That is what
    makes filling and validating composable: :func:`apply_defaults` fills every field, so a
    configuration on its way to a tool carries ``None`` for everything nobody configured and has to
    survive its own type, enum and expression checks.

    ``required`` and ``non_empty`` are checked separately and answer different questions.
    ``required`` refuses ``None`` -- did a value arrive at all -- and is satisfied by ``[]``,
    ``""`` or ``{}``, each of which is a value. ``non_empty`` refuses exactly those empty values on
    a field that declares it, whether or not the field is also ``required``, because knowing a
    value arrived says nothing about whether it carries anything.
    """
    _check_level(template, config, prefix="")


def _check_level(template: Mapping[str, ConfigField], config: Mapping[str, Any], prefix: str) -> None:
    """One level of a configuration against one level of a template, reporting paths under ``prefix``."""
    _refuse_key_that_is_not_a_name(config, prefix)

    unknown = set(config) - set(template)
    if unknown:
        raise ConfigError(f"unknown config field(s): {', '.join(sorted(_joined(prefix, name) for name in unknown))}")

    for name, spec in template.items():
        path = _joined(prefix, name)
        _refuse_unknown_type(path, spec)
        if spec.required and config.get(name) is None:
            raise ConfigError(f"missing required config field {path!r}")

    for name, spec in template.items():
        if name in config:
            _check_value(_joined(prefix, name), spec, config[name])


def _refuse_key_that_is_not_a_name(config: Mapping[str, Any], prefix: str) -> None:
    """Refuse a configuration key that is not a string, before anything tries to report it.

    A key like this comes from a definition file. The author wrote ``on: period``, the loader
    turned the bare word into ``True``, and the key stopped being a name. So this is its own
    refusal rather than an unknown field. An unknown field is a typo, and the fix is a different
    word. This is a name the file ate, and the fix is a pair of quotes.

    The message carries the type as well as the key, because ``True`` and ``"True"`` print the
    same and the difference between them is the whole problem.
    """
    for key in config:
        if isinstance(key, str):
            continue
        where = f"config field {prefix!r} has" if prefix else "config has"
        raise ConfigError(
            f"{where} a key that is not a name: {key!r} ({type(key).__name__}). YAML turns bare "
            f"words like on/no/true into other types -- quote the key in the file."
        )


def _check_value(path: str, spec: ConfigField, value: Any) -> None:
    """One value against one field spec, walking into it wherever the spec describes what is inside."""
    _refuse_unknown_type(path, spec)
    if value is None:
        return

    if not _TYPE_CHECKS[spec.type](value):
        raise ConfigError(f"config field {path!r} must be of type {spec.type!r}, got {type(value).__name__}")

    if spec.non_empty and spec.type in _SIZED_TYPES and len(value) == 0:
        raise ConfigError(f"config field {path!r} must not be empty")

    if spec.type == "array" and spec.items is not None:
        for index, item in enumerate(value):
            _check_value(f"{path}[{index}]", spec.items, item)
        return

    if spec.type == "object" and spec.fields is not None:
        _check_level(spec.fields, value, prefix=path)
        return

    _check_no_placeholder(path, value)

    if spec.enum is not None and value not in spec.enum:
        raise ConfigError(f"config field {path!r} has value {value!r}, but must be one of: {', '.join(spec.enum)}")

    if spec.type == "expression":
        try:
            compile_expr(value, columns=None)
        except ExpressionError as error:
            raise ConfigError(f"config field {path!r} does not compile: {error}") from error


def _refuse_unknown_type(path: str, spec: ConfigField) -> None:
    """Refuse a declared type nothing knows how to check, rather than checking nothing."""
    if spec.type not in FIELD_TYPES:
        raise ConfigError(
            f"config field {path!r} declares unknown type {spec.type!r}; the types a template may "
            f"declare are: {', '.join(sorted(FIELD_TYPES))}"
        )


def _check_no_placeholder(path: str, value: Any) -> None:
    """Refuse ``${...}`` anywhere inside ``value``, walking into an array or object at any depth."""
    if isinstance(value, str):
        if "${" in value:
            raise ConfigError(f"config field {path!r} still holds an unresolved parameter placeholder: {value!r}")
    elif isinstance(value, list):
        for item in value:
            _check_no_placeholder(path, item)
    elif isinstance(value, dict):
        for item in value.values():
            _check_no_placeholder(path, item)


def with_base_dir(template: Mapping[str, ConfigField], config: Mapping[str, Any], base_dir: Path) -> dict[str, Any]:
    """``config`` with every relative path in a ``file_path`` field joined onto ``base_dir``.

    An absolute path is left as it is, and so is a value holding a ``{`` token such as
    ``{scratch}``, which something else resolves. Only text values are touched.
    """
    resolved = dict(config)
    for name, spec in template.items():
        value = resolved.get(name)
        if spec.file_path and isinstance(value, str) and value and "{" not in value and not Path(value).is_absolute():
            resolved[name] = str(base_dir / value)
    return resolved
