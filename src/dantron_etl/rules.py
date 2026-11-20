"""Fill in ``${rules.*}`` references in a step's configuration.

A pipeline file can name a value from a rules file instead of writing it out, as ``${rules.name}``.
The loader replaces each reference with the rule's value before it checks the configuration. A
reference that is a whole field takes the rule's own value and type. One inside other text is
written into it, spelled for the kind of field it sits in.
"""

import re
from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

from .config import ConfigField
from .errors import PipelineError
from .expr import escape_text, quote_text

__all__ = ["resolve_rules"]


# Any ``${...}`` reference in a string.
_PLACEHOLDER: re.Pattern[str] = re.compile(r"\$\{([^}]*)\}")


# ---------------------------------------------------------------------------
# Parameter resolution
# ---------------------------------------------------------------------------


def resolve_rules(
    template: Mapping[str, ConfigField],
    config: Mapping[str, Any],
    rules: Mapping[str, Any],
    step_id: str,
) -> dict[str, Any]:
    """Return a copy of ``config`` with every ``${rules.*}`` reference filled in, at any depth.

    - ``template``: the tool's configuration template, which says what kind each field is.
    - ``rules``: the rule values by name.
    - ``step_id``: named in any refusal.

    A reference that is the whole field takes the rule's value and type, so a list stays a list.
    A reference inside other text is written in: in an expression field it is spelled as the
    expression language writes a value (``'O''Brian'``, ``('A', 'B')``, ``date'2026-04-01'``); in
    any other field it is written as plain text. Refuses an undefined rule, a namespace other than
    ``rules``, and a value that has no single way to be written where it sits.

    Why: docs/design/loading-and-running.md#rule-references
    """
    return {key: _resolve_value(value, template.get(key), rules, step_id) for key, value in config.items()}


def _resolve_value(value: Any, spec: ConfigField | None, rules: Mapping[str, Any], step_id: str) -> Any:
    """Fill in references in one value, walking down the value and its template entry together.

    ``spec`` is ``None`` where the template says nothing, and references there are written as plain text.
    """
    if isinstance(value, str):
        return _resolve_text(value, spec, rules, step_id)
    if isinstance(value, list):
        element = spec.items if spec is not None and spec.type == "array" else None
        return [_resolve_value(item, element, rules, step_id) for item in value]
    if isinstance(value, dict):
        declared = spec.fields if spec is not None and spec.type == "object" else None
        members: Mapping[str, ConfigField] = declared if declared is not None else {}
        return {key: _resolve_value(item, members.get(key), rules, step_id) for key, item in value.items()}
    return value


def _resolve_text(text: str, spec: ConfigField | None, rules: Mapping[str, Any], step_id: str) -> Any:
    """Fill in references in one string. A string that is only a reference becomes the rule's value.

    Each reference is spelled for where it sits. Whether it is inside the author's quotes is read
    from the string as written, not from the text already filled in.
    """
    if "${" not in text:
        return text
    whole = _PLACEHOLDER.fullmatch(text)
    if whole is not None:
        return _rule_value(whole.group(1), rules, step_id)
    is_expression = spec is not None and spec.type == "expression"

    def written(match: re.Match[str]) -> str:
        value = _rule_value(match.group(1), rules, step_id)
        rule = match.group(1).removeprefix("rules.")
        if not is_expression:
            return _as_text(value, rule=rule, step_id=step_id, where="into this field's text")
        return _as_expression(value, inside_literal=_inside_quotes(text, match.start()), rule=rule, step_id=step_id)

    return _PLACEHOLDER.sub(written, text)


def _inside_quotes(text: str, offset: int) -> bool:
    """Say whether ``offset`` is inside quotes the author wrote.

    A quote inside a column name in square brackets, like ``[client's name]``, does not count. A
    doubled quote inside text is one quote.
    """
    inside_text = False
    inside_brackets = False
    for char in text[:offset]:
        if inside_brackets:
            inside_brackets = char != "]"
        elif char == "'":
            inside_text = not inside_text
        elif char == "[" and not inside_text:
            inside_brackets = True
    return inside_text


def _kind(value: Any) -> str:
    """Describe a rule value's type in the words a refusal uses."""
    if value is None:
        return "a null"
    if isinstance(value, bool):
        return "a truth value"
    if isinstance(value, (list, tuple)):
        return "a list"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, float):
        return "a decimal"
    if isinstance(value, datetime):
        return "a timestamp"
    return f"a {type(value).__name__}"


def _as_text(value: Any, *, rule: str, step_id: str, where: str) -> str:
    """Write a rule value as plain text: text as it is, a whole number in digits, a date as YYYY-MM-DD.

    Anything else is refused, because it has no single text form. A YAML ``no`` would otherwise
    become ``False``.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return value.isoformat()
    raise PipelineError(
        f"step {step_id!r} writes rule {rule!r} {where}, but the rule's value is {_kind(value)} "
        f"({value!r}), which has no single text form. Quote the value in the rules file to state the "
        f"text you mean. YAML reads bare words like yes, no and null as other types, and a number with "
        f"a decimal point as a decimal."
    )


def _as_expression(value: Any, *, inside_literal: bool, rule: str, step_id: str) -> str:
    """Write a rule value into an expression in the expression language's own spelling.

    Inside the author's quotes (``'${rules.code}'``) the value is written as escaped text; a list,
    a truth value or a null is refused there. Outside quotes: text is quoted, a number written as
    is, ``true``/``false``/``null``, a date as ``date'YYYY-MM-DD'``, a list as ``(a, b)``. An empty
    list, a nested list and a mapping are refused.
    """
    if inside_literal:
        if isinstance(value, (list, tuple)):
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r} inside quotes, where this language reads text, but "
                f"the rule's value is a list. To match any value in the list, write it without quotes "
                f"after 'in', as in [code] in ${{rules.{rule}}}"
            )
        if value is None or isinstance(value, bool):
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r} inside quotes, where this language reads "
                f"text, but the rule's value is {value!r} -- a truth value or a null, which this "
                f"language spells without quotes. YAML reads bare words like yes/no/on/off/null "
                f"that way too: quote the value in the rules file to state the text you meant."
            )
        return escape_text(_as_text(value, rule=rule, step_id=step_id, where="inside quotes in an expression"))
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        if not value:
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r}, which is an empty list, so 'in' would match "
                f"nothing and 'not in' everything. Give the rule at least one value"
            )
        if any(isinstance(item, (list, tuple, Mapping)) for item in value):
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r}, a list holding another list or a mapping; a list "
                f"of values to match holds text, numbers or dates only"
            )
        elements = (_as_expression(item, inside_literal=False, rule=rule, step_id=step_id) for item in value)
        return "(" + ", ".join(elements) + ")"
    if isinstance(value, str):
        return quote_text(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return f"date'{value.isoformat()}'"
    if isinstance(value, (int, float)):
        return str(value)
    raise PipelineError(
        f"step {step_id!r} writes rule {rule!r} into an expression, but the rule's value is {_kind(value)} "
        f"({value!r}), which this language has no way to write. A rule written into an expression holds "
        f"text, a number, a date, true, false, null, or a list of those"
    )


def _rule_value(reference: str, rules: Mapping[str, Any], step_id: str) -> Any:
    """Return the value ``${reference}`` names. Refuse an undefined rule or a namespace other than ``rules``."""
    if not reference.startswith("rules."):
        raise PipelineError(
            f"step {step_id!r} references '${{{reference}}}', but 'rules' is the only namespace a "
            f"pipeline configuration may reference"
        )
    rule_name = reference[len("rules.") :]
    if rule_name not in rules:
        available = ", ".join(sorted(rules)) if rules else "none"
        raise PipelineError(
            f"step {step_id!r} references undefined rule {rule_name!r}; rules on offer are: {available}"
        )
    return rules[rule_name]
