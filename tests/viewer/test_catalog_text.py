"""Tests for the catalog as terminal text: ``describe(name)``, ``tools_text()`` and ``functions_text()``.

These are what ``help <tool>``, ``tools`` and ``functions`` print. A reader at a terminal builds a
step from ``describe`` alone, so it has to carry everything the registry declares about a tool, and
its worked example has to be YAML that pastes straight into a pipeline file.

Everything expected is read from the registry and the function table, never listed here, so a tool
or a function added later is covered on the day it arrives. The tests assert properties, not
wording, with a few conventions the implementation must follow so the text can be checked:

* **Example block.** ``describe`` prints a line that is exactly ``Example settings (YAML):`` (after
  stripping). The YAML is the run of lines straight after it that start with four spaces; the first
  line that does not (a blank line included) ends the block. With four spaces taken off each line,
  ``yaml.safe_load`` of the block gives the first worked example's configuration exactly.
* **Wrapped prose.** Descriptions, notes and summaries may be wrapped to the terminal. They are
  compared with all runs of whitespace folded to one space, so wrap without breaking words at
  hyphens or in the middle (``textwrap`` with ``break_on_hyphens=False, break_long_words=False``).
* **Settings rows.** Some line naming a required field also holds the word ``required``. Some line
  naming a field that has a default holds the word ``default`` and the default value, and does not
  hold the word ``required``.
* **Tools list.** One line per tool whose first word is the tool's name. Each category is named on a
  line whose first word is the category, before that category's tools.
* **Catalog page.** Inside a tool's card (from ``id="tool-<name>"`` to the next ``id="tool-``), the
  first ``<pre>`` whose class list includes ``yaml`` holds the first worked example's configuration
  as YAML, HTML-escaped.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any

import pytest
import yaml

from dantron_etl.authoring import REGISTRY
from dantron_etl.catalog import CATEGORY_ORDER, catalog
from dantron_etl.contract import catalog_input_ports
from dantron_etl.expr import ALLOWED_FUNCTIONS
from dantron_etl.expr.functions import DEFINITIONS

EXAMPLE_MARKER = "Example settings (YAML):"
INDENT = "    "

TOOL_NAMES = sorted(tool.name for tool in REGISTRY.tools())


def _describe(name: str) -> str:
    from dantron_etl.catalog import describe

    return describe(name)


def _tool(name: str) -> Any:
    return next(tool for tool in REGISTRY.tools() if tool.name == name)


def _plain(value: Any) -> Any:
    """A configuration as plain dicts and lists, the shape ``yaml.safe_load`` gives back."""
    if isinstance(value, Mapping):
        return {key: _plain(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(inner) for inner in value]
    return value


def _fold(text: str) -> str:
    return " ".join(text.split())


def _example_block(text: str) -> str:
    """The YAML block that follows the example marker line, with its indent taken off."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == EXAMPLE_MARKER]
    assert starts, f"no line {EXAMPLE_MARKER!r} in the text"
    block = []
    for line in lines[starts[0] + 1 :]:
        if not line.startswith(INDENT):
            break
        block.append(line[len(INDENT) :])
    assert block, "the example marker is not followed by an indented YAML block"
    return "\n".join(block)


def _before_example(text: str) -> str:
    return text.split(EXAMPLE_MARKER, 1)[0]


def _after_example(text: str) -> str:
    return text.split(EXAMPLE_MARKER, 1)[1]


def _has_word(line: str, word: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(word)}(?![\w])", line) is not None


def _first_word(line: str) -> str:
    words = line.split()
    return words[0].rstrip(":") if words else ""


def _default_forms(value: Any) -> set[str]:
    """The ways a default may be written: Python's ``str`` or YAML's own spelling."""
    yaml_form = yaml.safe_dump(value, default_flow_style=True).strip()
    yaml_form = yaml_form.removesuffix("...").strip()
    return {str(value), yaml_form}


class TestDescribe:
    """``describe(name)``: one tool, everything the registry says about it, as terminal text."""

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_names_the_tool_its_category_ports_fields_and_notes(self, name: str) -> None:
        tool = _tool(name)
        helped = tool.help()
        text = _describe(name)
        folded = _fold(text)

        assert _has_word(text, tool.name)
        assert _has_word(text, tool.category)
        assert _fold(tool.description) in folded
        assert _fold(helped.detailed_description) in folded
        for port in (*catalog_input_ports(tool), *tool.output_ports):
            assert _has_word(text, port), f"{name}: port {port!r} is not mentioned"
        for field in tool.config_template():
            assert _has_word(_before_example(text), field), f"{name}: field {field!r} is not in the settings"
        for note in helped.notes:
            assert _fold(note) in folded, f"{name}: a note is missing: {note[:60]!r}"

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_example_yaml_loads_back_to_the_first_worked_example(self, name: str) -> None:
        tool = _tool(name)
        expected = _plain(tool.help().examples[0].configuration)

        loaded = yaml.safe_load(_example_block(_describe(name)))

        assert loaded == expected

    @pytest.mark.parametrize("name", TOOL_NAMES)
    def test_example_shows_every_port_and_column_of_its_input_and_output_rows(self, name: str) -> None:
        tool = _tool(name)
        example = tool.help().examples[0]
        shown = _after_example(_describe(name))

        for ports in (example.inputs, example.outputs):
            for port, frame in ports.items():
                assert _has_word(shown, port), f"{name}: example port {port!r} is not shown"
                for column in frame.columns:
                    assert _has_word(shown, column), f"{name}: example column {column!r} is not shown"

    def test_required_fields_say_so_and_defaulted_fields_show_their_default(self) -> None:
        checked_required = checked_default = 0
        for tool in REGISTRY.tools():
            lines = _before_example(_describe(tool.name)).splitlines()
            for field, spec in tool.config_template().items():
                naming = [line for line in lines if _has_word(line, field)]
                if spec.required:
                    assert any(_has_word(line, "required") for line in naming), (
                        f"{tool.name}.{field} is required and no line naming it says so"
                    )
                    checked_required += 1
                elif spec.default is not None:
                    forms = _default_forms(spec.default)
                    assert any(
                        _has_word(line, "default")
                        and not _has_word(line, "required")
                        and any(form in line for form in forms)
                        for line in naming
                    ), f"{tool.name}.{field} defaults to {spec.default!r} and no line shows it"
                    checked_default += 1
        assert checked_required and checked_default

    def test_every_allowed_value_is_listed_in_the_settings(self) -> None:
        checked = 0
        for tool in REGISTRY.tools():
            settings = _before_example(_describe(tool.name))
            for field, spec in tool.config_template().items():
                for value in spec.enum or ():
                    assert _has_word(settings, str(value)), f"{tool.name}.{field}: allowed value {value!r} not listed"
                    checked += 1
        assert checked

    def test_unknown_name_is_refused_naming_the_close_match(self) -> None:
        with pytest.raises(ValueError) as caught:
            _describe("joim")

        message = str(caught.value)
        assert "joim" in message
        assert _has_word(message, "join")

    def test_unknown_name_with_no_close_match_is_still_a_plain_refusal(self) -> None:
        # The mirror: nothing close to suggest must not fall through to a KeyError or an empty text.
        with pytest.raises(ValueError) as caught:
            _describe("zzqxw_nothing_like_it")

        assert "zzqxw_nothing_like_it" in str(caught.value)


class TestToolsText:
    """``tools_text()``: every tool on one line, grouped by category in ``CATEGORY_ORDER``."""

    def test_every_tool_is_listed_exactly_once(self) -> None:
        from dantron_etl.catalog import tools_text

        firsts = [_first_word(line) for line in tools_text().splitlines()]

        for name in TOOL_NAMES:
            assert firsts.count(name) == 1, f"{name} is listed {firsts.count(name)} times"

    def test_tools_are_grouped_under_their_category_in_category_order(self) -> None:
        from dantron_etl.catalog import tools_text

        category_of = {tool.name: tool.category for tool in REGISTRY.tools()}
        rank = {category: i for i, category in enumerate(CATEGORY_ORDER)}
        current: str | None = None
        seen_order: list[str] = []
        for line in tools_text().splitlines():
            word = _first_word(line)
            if word in rank and word not in category_of:
                current = word
                seen_order.append(word)
            elif word in category_of:
                assert current == category_of[word], f"{word} is listed under {current!r}"

        assert seen_order == sorted(seen_order, key=rank.__getitem__)
        assert set(seen_order) == set(category_of.values()) & set(rank)


class TestFunctionsText:
    """``functions_text()``: every expression function with its summary and example."""

    def test_every_function_is_listed_with_its_summary_and_example(self) -> None:
        from dantron_etl.catalog import functions_text

        text = functions_text()
        folded = _fold(text)
        definitions = {definition.name: definition for definition in DEFINITIONS}

        assert set(ALLOWED_FUNCTIONS) <= set(definitions)
        for name in ALLOWED_FUNCTIONS:
            definition = definitions[name]
            assert _has_word(text, name), f"{name} is not listed"
            assert definition.example in text, f"{name}: example {definition.example!r} is not shown"
            assert _fold(definition.summary) in folded, f"{name}: summary is not shown"


class TestCatalogPageYaml:
    """The catalog page shows each tool's first worked example configuration as YAML."""

    def test_each_card_shows_its_first_example_as_yaml_that_loads_back(self) -> None:
        from dantron_etl.viewer.catalog_page import build_page

        page = build_page(catalog())
        for tool in REGISTRY.tools():
            start = page.find(f'id="tool-{tool.name}"')
            assert start != -1, f"no card for {tool.name}"
            end = page.find('id="tool-', start + 1)
            card = page[start : end if end != -1 else len(page)]
            found = re.search(r'<pre[^>]*class="[^"]*\byaml\b[^"]*"[^>]*>(.*?)</pre>', card, re.DOTALL)
            assert found, f"{tool.name}: no YAML block on its card"

            loaded = yaml.safe_load(html.unescape(found.group(1)))

            assert loaded == _plain(tool.help().examples[0].configuration), tool.name
