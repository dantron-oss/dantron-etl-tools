"""Resolving ``${rules.*}`` references in a step's configuration.

A pipeline file can name a value from a rules file instead of writing it out, as ``${rules.name}``.
The loader replaces each reference with the rule's value before it checks the configuration. A
reference that is a whole field takes the rule's own value and type; one inside other text is
written into it, spelled for the kind of field it sits in.
"""

import re
from collections.abc import Mapping
from typing import Any

from .config import ConfigField
from .errors import PipelineError
from .expr import escape_text, quote_text

__all__ = ["resolve_rules"]


# Any ``${...}`` reference, wherever it sits in a string. Whether it is the whole field or text
# embedded inside a larger one is decided by the caller, matching it with ``fullmatch`` or ``sub``.
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
    """``config`` with every ``${rules.*}`` resolved, at any depth, before validation ever sees it.

    A reference that is the whole field takes the rule's own value and type, so a list-valued rule
    stays a list. That is what it does in every field, whatever the field is for.

    A reference sitting inside surrounding text is written into that text, and how it is written is
    the field's question rather than this loader's. ``template`` is what answers it. A field
    declared ``type="expression"`` gets the value in the expression language's own spelling --
    quoted, escaped, a list parenthesised as ``('DRAFT', 'SUBMITTED')`` because square brackets in
    an expression mean a column -- since the person editing a pipeline file is stating a value, not
    writing code, and has no reason to know that ``O'Brian`` is spelled differently from
    ``Parkside``. Every other field gets the value and nothing else: a pattern, a path, a sheet
    name, a column name and a table name are none of them expressions, and a quote written into one
    of those is not an escape. It is a character to match, or a character in a file name.

    A field the template does not describe is written verbatim too, and is never refused here. A
    field nobody declared is a misspelled name, and ``validate_against_template`` is what says so;
    a refusal about rules would send the author to the wrong half of the file.

    The ``{scratch}`` token carries no dollar sign and is never touched here. It belongs to the
    tool boundary, not to this loader.
    """
    return {key: _resolve_value(value, template.get(key), rules, step_id) for key, value in config.items()}


def _resolve_value(value: Any, spec: ConfigField | None, rules: Mapping[str, Any], step_id: str) -> Any:
    """One value and the template entry describing it, walked down together.

    Together rather than separately, because the fields this distinction matters to are nested: a
    ladder's conditions, an aggregation's value and a derived column's name all sit two levels down,
    and that last one sits one element away from an expression wanting the opposite spelling.

    ``spec`` is ``None`` wherever the template stops -- an undeclared field, an array with no
    declared element, an object with no declared members -- and everything under it is verbatim.
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
    """One string: the rule's own value where the reference is the whole of it, or written into it.

    Each reference is spelled for where it sits, one at a time, because a field may hold two of them
    wanting opposite spellings -- one inside the author's quotes and one outside. The parity that
    answers that is read from the string as the author wrote it, never from the text produced so far.
    """
    if "${" not in text:
        return text
    whole = _PLACEHOLDER.fullmatch(text)
    if whole is not None:
        return _rule_value(whole.group(1), rules, step_id)
    is_expression = spec is not None and spec.type == "expression"

    def written(match: re.Match[str]) -> str:
        value = _rule_value(match.group(1), rules, step_id)
        if not is_expression:
            return value if isinstance(value, str) else str(value)
        return _as_expression(
            value,
            inside_literal=text[: match.start()].count("'") % 2 == 1,
            rule=match.group(1).removeprefix("rules."),
            step_id=step_id,
        )

    return _PLACEHOLDER.sub(written, text)


def _as_expression(value: Any, *, inside_literal: bool, rule: str, step_id: str) -> str:
    """One rule value, written into an expression field in the expression language's own spelling.

    Each element of a list is spelled by its own type: text is quoted and escaped, a number is not,
    because a number in quotes is text and text against a number column is refused.

    ``inside_literal`` says whether the reference sits inside quotes the author wrote themselves.
    ``[code] = '${rules.kept_code}'`` and ``[month] >= date'${rules.start_month}'`` are both written
    that way in real pipeline files, and both have to keep working. Inside, the value needs the
    escaping only; outside, it needs the quotes as well. The parity of ``'`` before the reference
    answers that, and stays right under doubling because every literal is balanced.

    A truth value and a null have a spelling in this language and get it -- but only outside a
    literal. Inside one, the author asked for text and the rules file holds none, so that is refused
    rather than written as ``'False'``, which compiles, matches nothing and reports success.
    """
    if value is None or isinstance(value, bool):
        if inside_literal:
            raise PipelineError(
                f"step {step_id!r} writes rule {rule!r} inside quotes, where this language reads "
                f"text, but the rule's value is {value!r} -- a truth value or a null, which this "
                f"language spells without quotes. YAML reads bare words like yes/no/on/off/null "
                f"that way too: quote the value in the rules file to state the text you meant."
            )
        if value is None:
            return "null"
        return "true" if value else "false"
    if inside_literal:
        return escape_text(str(value))
    if isinstance(value, (list, tuple)):
        elements = (_as_expression(item, inside_literal=False, rule=rule, step_id=step_id) for item in value)
        return "(" + ", ".join(elements) + ")"
    if isinstance(value, str):
        return quote_text(value)
    return str(value)


def _rule_value(reference: str, rules: Mapping[str, Any], step_id: str) -> Any:
    """The value ``${reference}`` names, or a refusal by name -- 'rules' is the only namespace."""
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
