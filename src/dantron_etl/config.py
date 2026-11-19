"""A tool's settings template, and the checks that hold a step's settings to it.

A tool describes its settings as a mapping of names to ``ConfigField``: a type, a description,
whether it is required, a default. This module checks settings against that template, fills in the
defaults, and finds the expressions the settings hold. It needs no frame, so all of it can run when
a pipeline file is loaded.

Why: docs/design/core.md#the-template-is-checked-at-the-boundary
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


# One check per field type. A boolean is not accepted as a number: in Python ``bool`` is an ``int``.
_TYPE_CHECKS: Final[dict[str, Any]] = {
    "string": lambda value: isinstance(value, str),
    "expression": lambda value: isinstance(value, str),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
}

# The field types a template may declare. Read from the checks above, so the two cannot differ.
FIELD_TYPES: Final[frozenset[str]] = frozenset(_TYPE_CHECKS)


# The types ``non_empty`` applies to. Zero is a value, so it means nothing on a number.
_SIZED_TYPES: Final[frozenset[str]] = frozenset({"string", "expression", "array", "object"})


@dataclass(frozen=True, slots=True)
class ConfigField:
    """One setting in a tool's template.

    ``ConfigField("string", "The column to read.", required=True)``

    - ``type``: one of :data:`FIELD_TYPES`. ``"expression"`` is a string the expression language
      compiles.
    - ``required``: refuses a missing value or ``None``. An empty ``[]``, ``""`` or ``{}`` still
      counts as a value.
    - ``non_empty``: refuses an empty ``[]``, ``""`` or ``{}``.
    - ``default``: filled in when the setting is not written. Never filled inside ``items`` or
      ``fields``.
    - ``enum``: the only values allowed.
    - ``items``: the shape of each element of an ``array``. ``fields``: the members of an
      ``object``. With these, settings nested in a list of small objects are checked too.

    Why: docs/design/core.md#nested-fields
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
    """Return the paths of the fields in ``template`` that hold an expression, in template order.

    A nested field is named by its path: ``"rules[].when"`` is the ``when`` member of each element
    of the ``rules`` array. The path has no index, because a template has no elements. For the
    expressions a step's settings actually hold, use :func:`expression_values`.
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
    """Return every expression ``config`` holds, as ``(path, text)`` pairs in template order.

    Paths carry the index, such as ``rules[0].when``, so a message can tell a person exactly
    which expression to fix. The loader uses this to check column names before any row is read.
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
    """Return a copy of ``config`` with every field of ``template`` present.

    A field not written gets its default, or ``None`` when it has none. So a tool can test
    ``config["x"] is None`` for any field. Nothing is filled inside an array or an object.

    Why: docs/design/core.md#every-field-is-filled
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
    """Check ``config`` against ``template``. Returns nothing; raises :class:`ConfigError`.

    Refuses: an unknown field; a required field that is missing or ``None``; a value of the wrong
    type; an empty value on a ``non_empty`` field; a value outside ``enum``; an expression that
    does not compile (syntax only, since no columns are known here); a ``${...}`` placeholder left
    in any string; a key that is not a string; and a field whose declared type is unknown.

    ``None`` is accepted for an optional field, so settings filled by :func:`apply_defaults` pass.
    Nested values are checked by the same rules and named by their path, such as
    ``rules[0].when``.

    Why: docs/design/core.md#what-the-template-check-refuses
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
    """Refuse a settings key that is not a string, such as YAML's ``on`` read as ``True``.

    Why: docs/design/core.md#a-key-yaml-turned-into-true
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
    """Return a copy of ``config`` with each relative path in a ``file_path`` field joined onto
    ``base_dir``.

    Absolute paths, values holding a ``{`` token such as ``{scratch}``, and values that are not
    text are left as they are.
    """
    resolved = dict(config)
    for name, spec in template.items():
        value = resolved.get(name)
        if spec.file_path and isinstance(value, str) and value and "{" not in value and not Path(value).is_absolute():
            resolved[name] = str(base_dir / value)
    return resolved
