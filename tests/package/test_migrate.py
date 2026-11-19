"""The migration tool: rewriting a pipeline file written before the renames.

The library now accepts only the new names. A file written earlier must be migrated, and the
migration has to edit the text in place: comments, layout, quoting and ``why`` text stay exactly as
they were, and only the names that were renamed change. A name is renamed by where it sits (which
tool, which field), never by searching the text, so most tests here pair a rename with its mirror:
the same word somewhere it must stay.

The round-trip test checks the result against a small renamer written in this file. It walks the
parsed document as plain dicts and shares nothing with the module under test.
"""

import copy
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml

import dantron_etl.migrate as migrate_module
from dantron_etl import PipelineError, load_pipeline, read_pipeline
from dantron_etl.authoring import REGISTRY, Ports, run_tool
from dantron_etl.migrate import MigrationError, main, migrate_text
from dantron_etl.paths import project_root

PROJECT_ROOT = project_root(__file__)
MIGRATE_COMMAND = "python -m dantron_etl.migrate"


def _steps(text: str) -> str:
    """``text`` as entries of ``steps:``: each ``- id:`` two spaces in, config fields six."""
    return textwrap.indent(textwrap.dedent(text), "  ")


def _flow(steps: str) -> str:
    """A pipeline file: a name, one read step, then ``steps``."""
    head = "name: Snippet\nsteps:\n  - id: src\n    tool: read_csv\n    config: {path: in.csv}\n"
    return head + _steps(steps)


# ---------------------------------------------------------------------------
# An old-style document that uses every rename, and an independent renamer
# ---------------------------------------------------------------------------

OLD_FLOW = r"""# A small flow written before the renames.
# Comments like this one must survive the migration exactly.
name: Orders and parties

lanes:
  # The lane that reads.
  - id: source
    label: Source
    order: 1
  - {id: shaping, label: Shaping, order: 2}

groups:
  - id: reading
    title: Read the orders and the parties.

steps:
  - id: orders
    tool: read_csv
    lane: source
    group: reading
    why: >
      The orders as they arrived. Words like index, values and matched
      are plain words here and must not change.
    config:
      path: orders.csv

  - id: parties
    tool: read_csv
    lane: source
    group: reading
    config: {path: parties.csv}

  # Pairing. The keys were called left_on and right_on before.
  - id: pair_up
    tool: join
    lane: shaping
    why: |
      Pair each order with its party.
      left_on and right_on name the keys; pair_up.matched is the result.
    config:
      left_on: [party_code]   # the order's side
      right_on: [code]
    inputs:
      left: orders.rows
      right: parties.rows

  - id: big_only
    tool: filter
    lane: shaping
    config:
      predicate: "len(strip([code])) > 0"
    inputs: {rows: pair_up.matched}   # the pairs that found a party

  - id: banded
    tool: case_when
    lane: shaping
    config:
      rules:
        - when: "[amount] > ${rules.threshold}"
          then: "strip([code])"
      otherwise: "'small'"
      output: band
    inputs: {rows: big_only.true}

  - id: long
    tool: unpivot
    lane: shaping
    config:
      index: [code]
      into: field
      values: value
      values_type: Float64
      on_invalid: empty
    inputs: {rows: banded.rows}

  - id: wide
    tool: cross_tab
    lane: shaping
    config:
      index: [code]
      columns_from: field
      values: value
      aggregate: count_values
    inputs: {rows: long.rows}

  - id: span
    tool: date_diff
    lane: shaping
    config:
      start: "[opened]"
      end: "[closed]"
      unit: months
      endpoints: both
      output: months_open
    inputs: {rows: wide.rows}

  - id: elapsed_days
    tool: date_diff
    lane: shaping
    config: {start: "[opened]", end: "[closed]", unit: days, output: days_open}
    inputs: {rows: span.rows}

  - id: periods
    tool: expand_date_range
    lane: shaping
    config:
      start: "[opened]"
      end: "[closed]"
      interval: month
    inputs: {rows: elapsed_days.rows}

  - id: dated
    tool: assemble_date
    lane: shaping
    config:
      year: "[year]"
      month: "[month]"
      on_invalid: empty
      output: order_date
    inputs: {rows: periods.rows}

  - id: typed
    tool: cast_columns
    lane: shaping
    config:
      columns:
        - {column: amount, type: Float64}
      on_invalid: empty
    inputs: {rows: dated.rows}

  - id: totals
    tool: group_by
    lane: shaping
    config:
      by: [code]
      aggregations:
        - name: filled
          function: count_values
          value: "strip([note])"
    inputs: {rows: typed.rows}

  - id: unique
    tool: remove_duplicates
    lane: shaping
    config:
      columns: [code]
    inputs: {rows: totals.rows}

  - id: best
    tool: match_one_to_one
    lane: shaping
    config:
      left: order_id
      right: party_id
      priority:
        - {column: score, direction: desc}
    inputs: {candidates: unique.unique}

  - id: named
    tool: normalize_text
    lane: shaping
    config:
      source: "[name]"
      output: clean_name
      operations: [lower, strip_suffixes]
      strip_suffixes: [ltd, inc]
    inputs: {rows: best.accepted}

  - id: linked
    tool: connected_components
    lane: shaping
    config:
      left: party_code
      right: code
      node_column: party
    inputs:
      edges: named.rows

  - id: kept
    tool: select_columns
    lane: shaping
    config:
      columns: [index, values, party]
    inputs: {rows: linked.rows}

  - id: pieces
    tool: regex_extract_groups
    lane: shaping
    config:
      source: "[note]"
      pattern: 'strip\((?P<inner>\w+)\)'
    inputs: {rows: kept.rows}

  - id: figured
    tool: formula
    lane: shaping
    config:
      columns:
        - name: tidy_code
          value: "strip([code])"
    inputs: {rows: pieces.rows}

  - id: checked
    tool: assert_rows
    lane: shaping
    config:
      predicate: "[amount] >= 0"
      label: amounts are not negative
    inputs: {rows: figured.rows}

  - id: out
    tool: write_csv
    lane: shaping
    config:
      path: out.csv
      replace: true
    inputs: {rows: checked.rows}

  - id: book
    tool: write_workbook
    lane: shaping
    why: >
      Two tabs. The port called edges here is a sheet's port,
      not the old input of a grouping step.
    config:
      path: book.xlsx
      sheets:
        - {port: summary, sheet: Summary}
        - {port: edges, sheet: Edges}
      replace: true
    inputs:
      summary: wide.rows
      edges: span.rows

  - id: publish
    tool: publish_tables
    lane: shaping
    config:
      connection: TARGET_DB
      stamp: loaded_at
      tables:
        - {port: orders, table: orders, mode: merge, key: [code]}
    inputs: {orders: unique.unique}
"""

RULES = {"threshold": 100}

_TOOL_RENAMES = {"connected_components": "make_groups", "write_workbook": "write_excel"}

_FIELD_RENAMES: dict[str, dict[str, str]] = {
    "join": {"left_on": "left_keys", "right_on": "right_keys"},
    "unpivot": {"index": "by", "into": "names_to", "values": "values_to"},
    "cross_tab": {"index": "by", "columns_from": "names_from", "values": "values_from", "aggregate": "function"},
    "date_diff": {"endpoints": "counting"},
    "expand_date_range": {"interval": "unit"},
    "case_when": {"rules": "cases"},
    "remove_duplicates": {"columns": "by"},
    "publish_tables": {"stamp": "stamp_column"},
    "write_csv": {"replace": "overwrite"},
    "write_excel": {"replace": "overwrite_sheets"},
    "match_one_to_one": {"left": "left_key", "right": "right_key"},
    "normalize_text": {"strip_suffixes": "suffixes"},
    "make_groups": {"node_column": "record_column", "component_column": "group_column"},
}

# (tool, field after renaming) -> old value -> new value
_VALUE_RENAMES: dict[tuple[str, str], dict[str, str]] = {
    ("date_diff", "counting"): {"both": "inclusive"},
    ("date_diff", "unit"): {"months": "month", "days": "day"},
    ("assemble_date", "on_invalid"): {"empty": "set_null"},
    ("cast_columns", "on_invalid"): {"empty": "set_null"},
    ("unpivot", "on_invalid"): {"empty": "set_null"},
    ("cross_tab", "function"): {"count_values": "count_non_null", "n_unique": "count_distinct"},
}

# The expression fields this document puts a strip( call in, by (new) tool.
_EXPRESSION_FIELDS: dict[str, tuple[str, ...]] = {
    "filter": ("predicate",),
    "case_when": ("cases[].when", "cases[].then", "otherwise"),
    "group_by": ("aggregations[].value",),
    "formula": ("columns[].value",),
}


def _rename_strip_at(config: dict[str, Any], path: str) -> None:
    """Replace ``strip(`` with ``trim(`` in the string at ``path`` (``a[].b`` walks a list)."""
    if "[]." in path:
        array, member = path.split("[].", 1)
        for entry in config.get(array) or []:
            _rename_strip_at(entry, member)
    elif isinstance(config.get(path), str):
        config[path] = config[path].replace("strip(", "trim(")


def _renamed_step(step: dict[str, Any], tool_by_id: dict[str, str]) -> dict[str, Any]:
    step = copy.deepcopy(step)
    tool = _TOOL_RENAMES.get(step["tool"], step["tool"])
    step["tool"] = tool
    if "group" in step:
        step["container"] = step.pop("group")

    fields = _FIELD_RENAMES.get(tool, {})
    config = {fields.get(key, key): value for key, value in (step.get("config") or {}).items()}
    for (owner, field), values in _VALUE_RENAMES.items():
        if owner == tool and field in config:
            config[field] = values.get(config[field], config[field])
    if tool == "date_diff":
        config.setdefault("counting", "inclusive")
    if tool == "make_groups":
        config.setdefault("record_column", "node")
        config.setdefault("group_column", "component")
    if tool == "publish_tables":
        for table in config["tables"]:
            if "key" in table:
                table["keys"] = table.pop("key")
            if table.get("mode") == "merge":
                table.setdefault("when_incoming_is_null", "keep_published")
    if tool == "write_csv":
        config.setdefault("allow_empty", True)
    if tool == "write_excel":
        for sheet in config.get("sheets") or []:
            sheet.setdefault("allow_empty", True)
    if tool == "group_by":
        for aggregation in config["aggregations"]:
            if aggregation["function"] == "count_values":
                aggregation["function"] = "count_non_null"
            if aggregation["function"] == "n_unique":
                aggregation["function"] = "count_distinct"
    for path in _EXPRESSION_FIELDS.get(tool, ()):
        _rename_strip_at(config, path)
    if "config" in step:
        step["config"] = config

    if "inputs" in step:
        inputs: dict[str, str] = {}
        for port, source in step["inputs"].items():
            if tool == "make_groups" and port == "edges":
                port = "pairs"
            source_step, source_port = source.split(".", 1)
            if tool_by_id.get(source_step) == "join" and source_port == "matched":
                source = f"{source_step}.joined"
            inputs[port] = source
        step["inputs"] = inputs
    return step


def _renamed(document: dict[str, Any]) -> dict[str, Any]:
    """The old document with every rename applied, by a plain walk over dicts."""
    document = copy.deepcopy(document)
    if "groups" in document:
        document["containers"] = document.pop("groups")
    for lane in document.get("lanes") or []:
        if "label" in lane:
            lane["title"] = lane.pop("label")
    tool_by_id = {step["id"]: step["tool"] for step in document["steps"]}
    document["steps"] = [_renamed_step(step, tool_by_id) for step in document["steps"]]
    return document


def _comments(text: str) -> list[str]:
    """Every comment in ``text``, whole-line or trailing, in order."""
    return [line[line.index("#") :] for line in text.splitlines() if "#" in line]


def _why_blocks(text: str) -> list[list[str]]:
    """The raw lines of every ``why:`` value: the key line's value and each line indented under it."""
    lines = text.splitlines()
    blocks: list[list[str]] = []
    for number, line in enumerate(lines):
        match = re.match(r"^(\s*)why:(.*)$", line)
        if not match:
            continue
        indent = len(match.group(1))
        block = [match.group(2)]
        for following in lines[number + 1 :]:
            if following.strip() and len(following) - len(following.lstrip()) <= indent:
                break
            block.append(following)
        blocks.append(block)
    return blocks


class TestRoundTrip:
    """An old file, migrated, loads under the current library and means what it meant."""

    def test_the_migrated_document_equals_the_old_one_with_every_rename_applied(self) -> None:
        migration = migrate_text(OLD_FLOW)

        assert yaml.safe_load(migration.text) == _renamed(yaml.safe_load(OLD_FLOW))

    def test_the_migrated_document_loads_and_each_step_keeps_its_tool_and_wiring(self) -> None:
        migration = migrate_text(OLD_FLOW)

        pipeline = load_pipeline(yaml.safe_load(migration.text), rules=RULES)

        expected = _renamed(yaml.safe_load(OLD_FLOW))
        assert [step.id for step in pipeline.steps] == [step["id"] for step in expected["steps"]]
        assert [step.tool for step in pipeline.steps] == [step["tool"] for step in expected["steps"]]
        assert pipeline.step("big_only").inputs["rows"].port == "joined"
        assert set(pipeline.step("linked").inputs) == {"pairs"}
        assert [container.id for container in pipeline.containers] == ["reading"]
        assert [lane.title for lane in pipeline.lanes] == ["Source", "Shaping"]

    def test_comments_and_why_text_are_byte_identical(self) -> None:
        migration = migrate_text(OLD_FLOW)

        assert _comments(migration.text) == _comments(OLD_FLOW)
        assert _why_blocks(migration.text) == _why_blocks(OLD_FLOW)
        assert _why_blocks(OLD_FLOW), "the fixture must hold why blocks for this test to mean anything"

    def test_migrating_the_output_again_changes_nothing(self) -> None:
        once = migrate_text(OLD_FLOW)

        twice = migrate_text(once.text)

        assert twice.text == once.text
        assert twice.changes == []
        assert not twice.counts or sum(twice.counts.values()) == 0

    def test_a_document_with_only_current_names_comes_back_unchanged(self) -> None:
        current = migrate_text(OLD_FLOW).text

        assert migrate_text(current).text == current

    def test_each_change_names_its_rename_its_original_line_and_its_step(self) -> None:
        lines = OLD_FLOW.splitlines()
        left_on_line = next(n for n, line in enumerate(lines, start=1) if "left_on: [party_code]" in line)
        groups_line = lines.index("groups:") + 1
        port_line = next(n for n, line in enumerate(lines, start=1) if "pair_up.matched}" in line)

        migration = migrate_text(OLD_FLOW)

        by_rename = {change.rename: change for change in migration.changes}
        assert (by_rename["join.left_on -> left_keys"].line, by_rename["join.left_on -> left_keys"].step) == (
            left_on_line,
            "pair_up",
        )
        assert (by_rename["groups -> containers"].line, by_rename["groups -> containers"].step) == (groups_line, None)
        assert by_rename["port join.matched -> joined"].line == port_line
        assert migration.counts["join.left_on -> left_keys"] == 1
        assert migration.counts["strip() -> trim()"] == 4
        assert sum(migration.counts.values()) == len(migration.changes)


# ---------------------------------------------------------------------------
# The same word, renamed in one place and left alone in another
# ---------------------------------------------------------------------------


class TestRenamedByWhereItSits:
    """Each rename is by tool and path. The same word elsewhere stays byte-identical."""

    def test_values_becomes_values_to_under_unpivot_and_values_from_under_cross_tab(self) -> None:
        old = _flow("""\
          - id: long
            tool: unpivot
            config:
              index: [code]
              into: field
              values: value   # the measure
            inputs: {rows: src.rows}
          - id: wide
            tool: cross_tab
            config:
              index: [code]
              columns_from: field
              values: value
              aggregate: "count_values"
            inputs: {rows: long.rows}
        """)
        expected = _flow("""\
          - id: long
            tool: unpivot
            config:
              by: [code]
              names_to: field
              values_to: value   # the measure
            inputs: {rows: src.rows}
          - id: wide
            tool: cross_tab
            config:
              by: [code]
              names_from: field
              values_from: value
              function: "count_non_null"
            inputs: {rows: long.rows}
        """)

        assert migrate_text(old).text == expected

    def test_columns_becomes_by_under_remove_duplicates_and_stays_under_select_columns_and_formula(self) -> None:
        old = _flow("""\
          - id: unique
            tool: remove_duplicates
            config:
              columns: [code]
            inputs: {rows: src.rows}
          - id: kept
            tool: select_columns
            config:
              columns: [index, values]
            inputs: {rows: unique.unique}
          - id: figured
            tool: formula
            config:
              columns:
                - {name: doubled, value: "[amount] * 2"}
            inputs: {rows: kept.rows}
        """)
        expected = old.replace("      columns: [code]", "      by: [code]")

        assert migrate_text(old).text == expected

    def test_rules_becomes_cases_under_case_when_and_a_rules_placeholder_stays(self) -> None:
        old = _flow("""\
          - id: banded
            tool: case_when
            config:
              rules:   # first match wins
                - {when: "[amount] > ${rules.threshold}", then: "'high'"}
              otherwise: "'${rules.values}'"
              output: band
            inputs: {rows: src.rows}
        """)
        expected = old.replace("      rules:   #", "      cases:   #")

        assert migrate_text(old).text == expected

    def test_label_becomes_title_in_a_lane_and_stays_on_assert_rows(self) -> None:
        old = textwrap.dedent("""\
            name: Snippet
            lanes:
              - id: main
                label: Main lane   # drawn first
                order: 1
              - {id: side, label: Side lane, order: 2}
            steps:
              - id: src
                tool: read_csv
                lane: main
                config: {path: in.csv}
              - id: checked
                tool: assert_rows
                lane: side
                config:
                  predicate: "[amount] >= 0"
                  label: amounts are not negative
                inputs: {rows: src.rows}
        """)
        expected = old.replace("    label: Main lane", "    title: Main lane").replace(
            "{id: side, label: Side lane", "{id: side, title: Side lane"
        )

        assert migrate_text(old).text == expected

    def test_left_and_right_are_renamed_only_under_match_one_to_one(self) -> None:
        old = _flow("""\
          - id: pair_up
            tool: join
            config: {keys: [code]}
            inputs:
              left: src.rows
              right: src.rows
          - id: best
            tool: match_one_to_one
            config:
              left: order_id
              right: party_id
              priority: [{column: score}]
            inputs: {candidates: pair_up.left_only}
          - id: linked
            tool: connected_components
            config:
              left: order_id
              right: party_id
              node_column: item
              component_column: cluster
            inputs: {edges: best.accepted}
        """)
        expected = _flow("""\
          - id: pair_up
            tool: join
            config: {keys: [code]}
            inputs:
              left: src.rows
              right: src.rows
          - id: best
            tool: match_one_to_one
            config:
              left_key: order_id
              right_key: party_id
              priority: [{column: score}]
            inputs: {candidates: pair_up.left_only}
          - id: linked
            tool: make_groups
            config:
              left: order_id
              right: party_id
              record_column: item
              group_column: cluster
            inputs: {pairs: best.accepted}
        """)

        assert migrate_text(old).text == expected

    def test_the_strip_suffixes_setting_is_renamed_and_the_operation_value_is_not(self) -> None:
        old = _flow("""\
          - id: named
            tool: normalize_text
            config:
              source: "[name]"
              output: clean_name
              operations:
                - lower
                - strip_suffixes
              strip_suffixes: [ltd, inc]   # trailing words to drop
            inputs: {rows: src.rows}
        """)
        expected = old.replace("      strip_suffixes: [ltd", "      suffixes: [ltd")

        assert migrate_text(old).text == expected
        assert "        - strip_suffixes\n" in migrate_text(old).text

    def test_matched_is_renamed_when_it_comes_from_a_join_and_not_from_a_filter(self) -> None:
        old = _flow("""\
          - id: pair_up
            tool: join
            why: Reads src twice. Its output used to be pair_up.matched.
            config: {keys: [code]}
            inputs: {left: src.rows, right: src.rows}
          - id: checked
            tool: filter
            config: {predicate: "[amount] > 0"}
            inputs: {rows: pair_up.matched}
          - id: out
            tool: write_csv
            config: {path: out.csv, allow_empty: true}
            inputs: {rows: checked.matched}
        """)
        expected = old.replace("{rows: pair_up.matched}", "{rows: pair_up.joined}")

        migration = migrate_text(old)

        assert migration.text == expected
        assert migration.counts["port join.matched -> joined"] == 1

    def test_strip_is_renamed_as_a_call_and_not_in_a_literal_a_column_name_or_a_pattern(self) -> None:
        old = _flow("""\
          - id: figured
            tool: formula
            why: Uses strip( on the code. Text like 'strip(x)' is left as it is.
            config:
              columns:
                - {name: tidy, value: "strip([code])"}
                - {name: said, value: "'strip(x)'"}
                - {name: copied, value: "[strip(x)]"}
                - {name: both, value: "concat(strip([a]), 'strip(')"}
            inputs: {rows: src.rows}
          - id: pieces
            tool: regex_extract_groups
            config:
              source: "[note]"
              pattern: 'strip\\((?P<inner>\\w+)\\)'
            inputs: {rows: figured.rows}
        """)
        expected = old.replace('"strip([code])"', '"trim([code])"').replace(
            "\"concat(strip([a]), 'strip(')\"", "\"concat(trim([a]), 'strip(')\""
        )

        migration = migrate_text(old)

        assert migration.text == expected
        assert migration.counts["strip() -> trim()"] == 2

    def test_a_flow_style_config_keeps_its_braces_and_quoting(self) -> None:
        old = _flow("""\
          - id: out
            tool: write_csv
            config: {path: 'out.csv', replace: true, delimiter: ";", allow_empty: true}
            inputs: {rows: src.rows}
        """)
        expected = old.replace("replace: true", "overwrite: true")

        assert migrate_text(old).text == expected


# ---------------------------------------------------------------------------
# Old defaults written out, so nothing changes
# ---------------------------------------------------------------------------


def _step_config(text: str, step_id: str) -> dict[str, Any]:
    document = yaml.safe_load(text)
    return next(step["config"] for step in document["steps"] if step["id"] == step_id)


class TestOldDefaultsWrittenOut:
    """Where an old default differs from the new one, the migration writes the old value in."""

    def test_a_block_date_diff_with_no_endpoints_gets_counting_inclusive_at_its_own_indentation(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: months
              output: months_open
            inputs: {rows: src.rows}
        """)

        migration = migrate_text(old)

        assert _step_config(migration.text, "span") == {
            "start": "[opened]",
            "end": "[closed]",
            "unit": "month",
            "counting": "inclusive",
            "output": "months_open",
        }
        assert "\n      counting: inclusive\n" in migration.text

    def test_a_flow_style_date_diff_with_no_endpoints_gets_counting_inclusive(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config: {start: '[a]', end: '[b]', unit: months, output: n}
            inputs: {rows: src.rows}
        """)

        migration = migrate_text(old)

        assert _step_config(migration.text, "span") == {
            "start": "[a]",
            "end": "[b]",
            "unit": "month",
            "counting": "inclusive",
            "output": "n",
        }
        assert "    config: {" in migration.text

    @pytest.mark.parametrize(("endpoints", "counting"), [("elapsed", "elapsed"), ("both", "inclusive")])
    def test_a_date_diff_that_named_its_endpoints_keeps_that_choice_under_counting(
        self, endpoints: str, counting: str
    ) -> None:
        old = _flow(f"""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: days
              endpoints: {endpoints}
              output: days_open
            inputs: {{rows: src.rows}}
        """)

        config = _step_config(migrate_text(old).text, "span")

        assert config["counting"] == counting
        assert config["unit"] == "day"
        assert "endpoints" not in config

    def test_a_grouping_step_relying_on_default_names_gets_them_written_and_its_output_is_unchanged(self) -> None:
        old = _flow("""\
          - id: linked
            tool: connected_components
            config:
              left: a
              right: b
            inputs: {edges: src.rows}
        """)

        config = _step_config(migrate_text(old).text, "linked")
        output = run_tool(
            REGISTRY.get("make_groups"), Ports({"pairs": pl.DataFrame({"a": [1, 2], "b": [2, 3]})}), config
        )["rows"]

        assert config == {"left": "a", "right": "b", "record_column": "node", "group_column": "component"}
        assert output.columns == ["node", "component"]

    def test_a_grouping_step_that_named_its_columns_keeps_its_own_names_under_the_new_keys(self) -> None:
        old = _flow("""\
          - id: linked
            tool: connected_components
            config: {left: a, right: b, node_column: item, component_column: cluster}
            inputs: {edges: src.rows}
        """)

        config = _step_config(migrate_text(old).text, "linked")
        output = run_tool(
            REGISTRY.get("make_groups"), Ports({"pairs": pl.DataFrame({"a": [1, 2], "b": [2, 3]})}), config
        )["rows"]

        assert config == {"left": "a", "right": "b", "record_column": "item", "group_column": "cluster"}
        assert output.columns == ["item", "cluster"]


# ---------------------------------------------------------------------------
# What cannot be migrated automatically
# ---------------------------------------------------------------------------


OLD_SINGLE_SHEET = _flow("""\
  # The old one-tab writer. It cannot be migrated by itself.
  - id: tab_out
    tool: write_excel
    why: >
      One tab of orders. Its replace flag
      is the old one-tab setting.
    config:
      path: orders.xlsx
      sheet: Orders
      replace: true
    inputs: {rows: src.rows}
""")


class TestSettingsNowStatedAreWrittenIn:
    """A setting that is now required, where the file alone says which value keeps a step's behaviour."""

    def test_a_block_merge_entry_gets_keep_published_at_its_own_indentation(self) -> None:
        old = _flow("""\
          - id: publish
            tool: publish_tables
            config:
              connection: TARGET_DB
              tables:
                - port: parties
                  table: parties
                  mode: merge
                  keys: [party]
                - port: totals
                  table: totals
                  mode: replace
            inputs: {parties: src.rows, totals: src.rows}
        """)

        migration = migrate_text(old)

        tables = _step_config(migration.text, "publish")["tables"]
        assert tables[0]["when_incoming_is_null"] == "keep_published"
        assert "when_incoming_is_null" not in tables[1], "a replace entry was given a merge setting"
        assert "\n          when_incoming_is_null: keep_published\n" in migration.text
        assert any("keep_published" in label and "today" in label for label in migration.counts), migration.counts

    def test_a_flow_style_merge_entry_gets_it_inside_its_braces(self) -> None:
        old = _flow("""\
          - id: publish
            tool: publish_tables
            config:
              connection: TARGET_DB
              tables:
                - {port: parties, table: parties, mode: merge, keys: [party]}
            inputs: {parties: src.rows}
        """)

        migration = migrate_text(old)

        assert "{port: parties, table: parties, mode: merge, keys: [party], when_incoming_is_null: keep_published}" in (
            migration.text
        )

    def test_every_file_writer_that_does_not_say_gets_allow_empty_true_which_is_what_it_did(self) -> None:
        old = _flow("""\
          - id: extract
            tool: write_csv
            config:
              path: out.csv
              overwrite: true
            inputs: {rows: src.rows}
          - id: strict
            tool: write_csv
            config: {path: strict.csv, allow_empty: false}
            inputs: {rows: src.rows}
          - id: pack
            tool: write_excel
            config:
              path: pack.xlsx
              sheets:
                - {port: one, sheet: Figures}
                - port: two
                  sheet: Checks
                  allow_empty: false
            inputs: {one: src.rows, two: src.rows}
        """)

        migration = migrate_text(old)

        assert _step_config(migration.text, "extract")["allow_empty"] is True
        assert _step_config(migration.text, "strict")["allow_empty"] is False, "a stated choice is kept"
        sheets = _step_config(migration.text, "pack")["sheets"]
        assert [sheet["allow_empty"] for sheet in sheets] == [True, False]
        assert "      allow_empty: true\n" in migration.text
        assert "{port: one, sheet: Figures, allow_empty: true}" in migration.text
        assert sum(migration.counts.values()) == 2

    def test_a_merge_entry_that_already_says_keeps_its_choice(self) -> None:
        old = _flow("""\
          - id: publish
            tool: publish_tables
            config:
              connection: TARGET_DB
              tables:
                - {port: parties, table: parties, mode: merge, keys: [party], when_incoming_is_null: overwrite}
            inputs: {parties: src.rows}
        """)

        migration = migrate_text(old)

        assert migration.text == old
        assert migration.changes == []


class TestReportedNotMigrated:
    """Things the tool sees and cannot safely change are reported, and left exactly as they are."""

    def test_an_old_single_sheet_write_excel_step_is_reported_and_left_byte_identical(self) -> None:
        migration = migrate_text(OLD_SINGLE_SHEET)

        assert migration.text == OLD_SINGLE_SHEET
        assert migration.changes == []
        assert any("tab_out" in entry for entry in migration.manual)

    def test_an_old_single_sheet_step_stays_untouched_while_the_rest_of_the_file_is_migrated(self) -> None:
        old = OLD_SINGLE_SHEET + _steps("""\
          - id: out
            tool: write_csv
            config: {path: out.csv, replace: true, allow_empty: true}
            inputs: {rows: tab_out.rows}
        """)
        expected = old.replace("{path: out.csv, replace: true, allow_empty: true}", "{path: out.csv, overwrite: true, allow_empty: true}")

        migration = migrate_text(old)

        assert migration.text == expected
        assert any("tab_out" in entry for entry in migration.manual)

    def test_a_renamed_value_held_in_a_rules_placeholder_is_reported_naming_the_step_and_line(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: "${rules.unit}"
              counting: elapsed
              output: n
            inputs: {rows: src.rows}
        """)
        unit_line = next(n for n, line in enumerate(old.splitlines(), start=1) if "rules.unit" in line)

        migration = migrate_text(old)

        assert migration.text == old
        reported = [entry for entry in migration.manual if "span" in entry]
        assert reported
        assert re.search(rf"\b{unit_line}\b", reported[0])


# ---------------------------------------------------------------------------
# Files that are not pipelines, and files that cannot be read
# ---------------------------------------------------------------------------


RULES_FLOW = textwrap.dedent("""\
    name: Rule uses
    steps:
      - id: orders
        tool: read_csv
        config: {path: orders.csv}
      - id: cleaned
        tool: formula
        config:
          columns:
            - {name: code_key, value: "${rules.clean_code}"}
            - {name: tidy, value: "${rules.tidy}"}
            - {name: tagged, value: "concat('x', '${rules.quoted}')"}
        inputs: {rows: orders.rows}
      - id: kept
        tool: filter
        config: {predicate: "${rules.tidy}"}
        inputs: {rows: cleaned.rows}
      - id: parts
        tool: regex_extract_groups
        config: {source: "[code]", pattern: "${rules.code_pattern}"}
        inputs: {rows: kept.true}
      - id: split_up
        tool: regex_extract_groups
        config: {source: "[code]", pattern: "${rules.both_ways}"}
        inputs: {rows: parts.rows}
      - id: both_check
        tool: filter
        config: {predicate: "${rules.both_ways}"}
        inputs: {rows: split_up.rows}
      - id: out
        tool: write_csv
        config: {path: out.csv, allow_empty: true}
        inputs: {rows: both_check.true}
""")

RULES_FILE = textwrap.dedent("""\
    # Rules for the orders flow.
    clean_code: "upper(strip([code]))"   # an expression, used whole in an expression field
    tidy: "strip([note]) != ''"
    quoted: "strip(x)"                   # written inside quotes in an expression: it is text there
    code_pattern: "strip(ped|ping)"      # a pattern, never an expression
    both_ways: "strip([code])"           # used whole in a predicate and as a pattern
    unused: "strip([other])"             # no pipeline given uses it
    threshold: 100
""")


class TestOtherKindsOfFile:
    """A rule's value is code only where a pipeline uses the rule as the whole of an expression field."""

    def _run(self, tmp_path: Path, *flags: str) -> tuple[int, Path, Path]:
        flow = tmp_path / "flow.yaml"
        rules = tmp_path / "rules.yaml"
        flow.write_text(RULES_FLOW, encoding="utf8")
        rules.write_text(RULES_FILE, encoding="utf8")
        return main([str(flow), str(rules), *flags]), flow, rules

    def test_a_rule_used_only_as_a_whole_expression_is_rewritten(self, tmp_path: Path) -> None:
        _, _, rules = self._run(tmp_path, "--write")

        text = rules.read_text(encoding="utf8")
        assert 'clean_code: "upper(trim([code]))"   # an expression, used whole in an expression field' in text
        assert "tidy: \"trim([note]) != ''\"" in text

    def test_a_rule_that_is_not_shown_to_be_an_expression_is_left_alone_and_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code, _, rules = self._run(tmp_path, "--write")
        printed = capsys.readouterr().out

        text = rules.read_text(encoding="utf8")
        for line in (
            'quoted: "strip(x)"',
            'code_pattern: "strip(ped|ping)"',
            'both_ways: "strip([code])"',
            'unused: "strip([other])"',
        ):
            assert line in text, f"{line} was changed"
        assert code == 1, "a file with something to check by hand is not done"
        report = printed[printed.index("rules.yaml") :]
        assert "not migrated automatically" in report
        assert "check by hand" in report
        for name, value in (
            ("quoted", "strip(x)"),
            ("code_pattern", "strip(ped|ping)"),
            ("both_ways", "strip([code])"),
            ("unused", "strip([other])"),
        ):
            assert re.search(rf"\b{name}\b.*{re.escape(value)}", report), f"{name} is not reported with its value"
        for name in ("clean_code", "tidy"):
            assert not re.search(rf"\b{name}\b.*check by hand", report), f"{name} was reported, not rewritten"

    def test_a_rules_file_on_its_own_changes_nothing_and_reports_every_call(self) -> None:
        migration = migrate_text(RULES_FILE)

        assert migration.text == RULES_FILE
        reported = " ".join(migration.manual)
        for name in ("clean_code", "tidy", "quoted", "code_pattern", "both_ways", "unused"):
            assert name in reported
        assert "threshold" not in reported

    @pytest.mark.parametrize("text", ["steps: [unclosed\n", "- a\n- b\n", "just words\n"])
    def test_text_that_is_not_a_yaml_mapping_is_refused(self, text: str) -> None:
        with pytest.raises(MigrationError):
            migrate_text(text)


# ---------------------------------------------------------------------------
# A file using a retired name, before and after
# ---------------------------------------------------------------------------

PAIRS = """\
# Pair orders with parties and write the pairs out.
name: Pairs
groups:
  - {id: reading, title: Read both sides.}
steps:
  - id: orders
    tool: read_csv
    group: reading
    config: {path: orders.csv}
  - id: parties
    tool: read_csv
    group: reading
    config: {path: parties.csv}
  - id: pair_up
    tool: join
    config:
      left_on: [party_code]   # the order's side
      right_on: [code]
    inputs: {left: orders.rows, right: parties.rows}
  - id: out
    tool: write_csv
    config: {path: out.csv, replace: true, allow_empty: true}
    inputs: {rows: pair_up.matched}
"""


class TestBeforeAndAfter:
    def test_a_file_with_a_retired_name_is_refused_with_the_new_name_and_loads_once_migrated(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_text(PAIRS, encoding="utf8")

        with pytest.raises(PipelineError) as refused:
            read_pipeline(path)
        message = str(refused.value)
        path.write_text(migrate_text(PAIRS).text, encoding="utf8")
        pipeline = read_pipeline(path)

        assert MIGRATE_COMMAND in message
        assert "'containers'" in message
        assert pipeline.step("out").inputs["rows"].port == "joined"
        assert dict(pipeline.step("pair_up").config) == {"left_keys": ["party_code"], "right_keys": ["code"]}


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


class TestCommandLine:
    def test_a_dry_run_reports_counts_by_rename_and_changes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_bytes(PAIRS.encode("utf8"))

        code = main([str(path)])
        printed = capsys.readouterr().out

        assert code == 1
        assert path.read_bytes() == PAIRS.encode("utf8")
        assert "pairs.yaml" in printed
        assert "join.left_on -> left_keys" in printed
        assert "port join.matched -> joined" in printed

    def test_a_dry_run_on_a_current_file_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_text(migrate_text(PAIRS).text, encoding="utf8")

        assert main([str(path)]) == 0

    def test_write_changes_the_file_and_a_second_run_finds_nothing_to_do(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_text(PAIRS, encoding="utf8")

        code = main([str(path), "--write"])

        assert code == 0
        assert path.read_text(encoding="utf8") == migrate_text(PAIRS).text
        assert main([str(path)]) == 0
        assert main([str(path), "--write"]) == 0
        assert path.read_text(encoding="utf8") == migrate_text(PAIRS).text

    def test_a_folder_means_every_yaml_file_under_it(self, tmp_path: Path) -> None:
        nested = tmp_path / "flows" / "deeper"
        nested.mkdir(parents=True)
        first, second, notes = tmp_path / "flows" / "a.yaml", nested / "b.yml", nested / "notes.txt"
        for path in (first, second, notes):
            path.write_text(PAIRS, encoding="utf8")

        code = main([str(tmp_path / "flows"), "--write"])

        assert code == 0
        assert first.read_text(encoding="utf8") == migrate_text(PAIRS).text
        assert second.read_text(encoding="utf8") == migrate_text(PAIRS).text
        assert notes.read_text(encoding="utf8") == PAIRS

    def test_an_unparseable_file_returns_two_is_named_and_left_alone_while_others_are_written(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        broken, good = tmp_path / "broken.yaml", tmp_path / "good.yaml"
        broken.write_text("steps: [unclosed\n  - left_on\n", encoding="utf8")
        good.write_text(PAIRS, encoding="utf8")

        code = main([str(broken), str(good), "--write"])
        printed = capsys.readouterr()

        assert code == 2
        assert broken.read_text(encoding="utf8") == "steps: [unclosed\n  - left_on\n"
        assert "broken.yaml" in printed.out + printed.err
        assert good.read_text(encoding="utf8") == migrate_text(PAIRS).text

    def test_port_map_writes_the_renamed_output_ports_as_json(self, tmp_path: Path) -> None:
        path, port_map = tmp_path / "pairs.yaml", tmp_path / "ports.json"
        path.write_text(PAIRS, encoding="utf8")

        main([str(path), "--port-map", str(port_map)])

        assert json.loads(port_map.read_text(encoding="utf8")) == {"tools": {"join": {"matched": "joined"}}}

    def test_the_python_renames_are_printed_as_guidance(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_text(migrate_text(PAIRS).text, encoding="utf8")

        main([str(path)])
        printed = capsys.readouterr().out

        for old, new in (("read_pipeline_file", "read_pipeline"), ("PipelineSpec", "Pipeline"), ("StepSpec", "Step")):
            assert old in printed
            assert new in printed

    def test_the_module_runs_as_a_command(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_text(PAIRS, encoding="utf8")
        environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

        result = subprocess.run(
            [sys.executable, "-m", "dantron_etl.migrate", str(path)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )

        assert result.returncode == 1, result.stderr
        assert "join.left_on -> left_keys" in result.stdout
        assert path.read_text(encoding="utf8") == PAIRS


# ---------------------------------------------------------------------------
# Cycle 2: cases found by review
# ---------------------------------------------------------------------------


class TestRenamedValuesByWhereTheySit:
    """A renamed value changes only in the field whose values were renamed."""

    def test_date_diff_unit_and_endpoints_change_and_an_output_column_named_months_stays(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: months
              endpoints: both
              output: months
            inputs: {rows: src.rows}
        """)
        expected = _flow("""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: month
              counting: inclusive
              output: months
            inputs: {rows: src.rows}
        """)

        assert migrate_text(old).text == expected

    def test_cast_columns_on_invalid_changes_and_a_column_named_empty_stays(self) -> None:
        old = _flow("""\
          - id: typed
            tool: cast_columns
            config:
              columns:
                - {column: empty, type: String}
              on_invalid: empty
            inputs: {rows: src.rows}
        """)
        expected = old.replace("      on_invalid: empty", "      on_invalid: set_null")

        assert migrate_text(old).text == expected

    def test_n_unique_becomes_count_distinct_and_the_report_says_nulls_are_now_skipped(self) -> None:
        old = _flow("""\
          - id: spread
            tool: group_by
            config:
              by: [n_unique]
              aggregations:
                - {name: n_unique, function: n_unique, value: "[term]"}
            inputs: {rows: src.rows}
          - id: wide
            tool: cross_tab
            config: {by: [code], names_from: period, values_from: label, function: n_unique}
            inputs: {rows: src.rows}
        """)
        expected = old.replace("function: n_unique, value", "function: count_distinct, value").replace(
            "label, function: n_unique}", "label, function: count_distinct}"
        )

        migration = migrate_text(old)

        assert migration.text == expected
        assert len(migration.changes) == 2
        assert all("skips nulls" in label for label in migration.counts), migration.counts

    def test_a_formula_columns_replace_flag_stays_while_write_csv_replace_becomes_overwrite(self) -> None:
        old = _flow("""\
          - id: figured
            tool: formula
            config:
              columns:
                - {name: amount, value: "[amount] * 2", replace: true}
            inputs: {rows: src.rows}
          - id: out
            tool: write_csv
            config: {path: out.csv, replace: true, allow_empty: true}
            inputs: {rows: figured.rows}
        """)
        expected = old.replace("{path: out.csv, replace: true, allow_empty: true}", "{path: out.csv, overwrite: true, allow_empty: true}")

        assert migrate_text(old).text == expected


class TestSharedAnchoredConfig:
    """A config written once under an anchor and reused by an alias is edited once, at the anchor."""

    def test_two_joins_sharing_an_anchored_flow_config_are_edited_once_at_the_anchor(self) -> None:
        old = _flow("""\
          - id: first
            tool: join
            config: &j {left_on: [a], right_on: [b]}
            inputs: {left: src.rows, right: src.rows}
          - id: second
            tool: join
            config: *j
            inputs: {left: src.rows, right: src.rows}
          - id: out_first
            tool: write_csv
            config: {path: first.csv, overwrite: true, allow_empty: true}
            inputs: {rows: first.joined}
          - id: out_second
            tool: write_csv
            config: {path: second.csv, overwrite: true, allow_empty: true}
            inputs: {rows: second.joined}
        """)
        expected = old.replace("&j {left_on: [a], right_on: [b]}", "&j {left_keys: [a], right_keys: [b]}")

        migration = migrate_text(old)
        pipeline = load_pipeline(yaml.safe_load(migration.text))

        assert migration.text == expected
        assert "    config: *j\n" in migration.text
        for step_id in ("first", "second"):
            assert dict(pipeline.step(step_id).config) == {"left_keys": ["a"], "right_keys": ["b"]}

    def test_two_date_diffs_sharing_an_anchored_block_config_rename_endpoints_once(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config: &d
              start: "[opened]"
              end: "[closed]"
              unit: day
              endpoints: both
              output: n
            inputs: {rows: src.rows}
          - id: again
            tool: date_diff
            config: *d
            inputs: {rows: span.rows}
        """)
        expected = old.replace("      endpoints: both\n", "      counting: inclusive\n")

        migration = migrate_text(old)

        assert migration.text == expected
        assert _step_config(migration.text, "again")["counting"] == "inclusive"

    def test_an_anchored_date_diff_config_with_no_endpoints_gets_counting_written_once_at_the_anchor(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config: &d
              start: "[opened]"
              end: "[closed]"
              unit: day
              output: n
            inputs: {rows: src.rows}
          - id: again
            tool: date_diff
            config: *d
            inputs: {rows: span.rows}
        """)
        expected = old.replace("      output: n\n", "      output: n\n      counting: inclusive\n")

        migration = migrate_text(old)

        assert migration.text == expected
        assert migration.text.count("counting: inclusive") == 1
        assert _step_config(migration.text, "span")["counting"] == "inclusive"
        assert _step_config(migration.text, "again")["counting"] == "inclusive"

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            # A join wants left_on renamed; a filter reusing the same text wants it left alone.
            (
                "tool: join\n    config: &c {left_on: [a], right_on: [b]}\n    inputs: {left: src.rows, right: src.rows}",
                "tool: filter\n    config: *c\n    inputs: {rows: src.rows}",
            ),
            # The old grouping step wants its old default names written in; the new one does not.
            (
                "tool: connected_components\n    config: &c {left: a, right: b}\n    inputs: {edges: src.rows}",
                "tool: make_groups\n    config: *c\n    inputs: {pairs: src.rows}",
            ),
        ],
        ids=["join_and_filter", "old_and_new_grouping_step"],
    )
    def test_an_anchor_shared_by_steps_needing_different_edits_is_refused(self, first: str, second: str) -> None:
        """The chosen behaviour: MigrationError, so the CLI names the file and leaves it as it was."""
        old = _flow("") + f"  - id: one\n    {first}\n  - id: two\n    {second}\n"

        with pytest.raises(MigrationError):
            migrate_text(old)


class TestRereadGuard:
    """When the edited text does not mean what the renames say, the file is refused and not written."""

    @pytest.fixture
    def misspelled_edit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Make the text edits spell one new key wrongly, as a bug in the editing would."""
        real_result = migrate_module._Editor.result

        def misspelled(editor: Any) -> str:
            return real_result(editor).replace("left_keys", "left_kyes", 1)

        monkeypatch.setattr(migrate_module._Editor, "result", misspelled)

    @pytest.mark.usefixtures("misspelled_edit")
    def test_a_wrong_edit_raises_migration_error(self) -> None:
        with pytest.raises(MigrationError):
            migrate_text(PAIRS)

    @pytest.mark.usefixtures("misspelled_edit")
    def test_a_wrong_edit_makes_write_return_two_and_leaves_the_file_unchanged(self, tmp_path: Path) -> None:
        path = tmp_path / "pairs.yaml"
        path.write_bytes(PAIRS.encode("utf8"))

        code = main([str(path), "--write"])

        assert code == 2
        assert path.read_bytes() == PAIRS.encode("utf8")


class TestGroupingStepColumnNames:
    """The old grouping step's default column names are written in only when the old tool is named."""

    def test_a_flow_style_grouping_config_gets_both_old_default_names_appended(self) -> None:
        old = _flow("""\
          - id: linked
            tool: connected_components
            config: {left: left, right: right}
            inputs: {edges: src.rows}
        """)
        expected = _flow("""\
          - id: linked
            tool: make_groups
            config: {left: left, right: right, record_column: node, group_column: component}
            inputs: {pairs: src.rows}
        """)

        migration = migrate_text(old)
        config = _step_config(migration.text, "linked")
        output = run_tool(
            REGISTRY.get("make_groups"), Ports({"pairs": pl.DataFrame({"left": [1, 2], "right": [2, 3]})}), config
        )["rows"]

        assert migration.text == expected
        assert output.columns == ["node", "component"]

    def test_a_half_migrated_make_groups_step_has_node_column_renamed_and_no_group_column_invented(self) -> None:
        old = _flow("""\
          - id: linked
            tool: make_groups
            config: {left: a, right: b, node_column: x}
            inputs: {pairs: src.rows}
        """)
        expected = old.replace("node_column: x", "record_column: x")

        migration = migrate_text(old)

        assert migration.text == expected
        reported = [entry for entry in migration.manual if "linked" in entry]
        assert reported
        assert "group_column" in reported[0]

    def test_an_old_grouping_step_naming_only_node_column_gets_group_column_component_written(self) -> None:
        old = _flow("""\
          - id: linked
            tool: connected_components
            config: {left: a, right: b, node_column: x}
            inputs: {edges: src.rows}
        """)
        expected = _flow("""\
          - id: linked
            tool: make_groups
            config: {left: a, right: b, record_column: x, group_column: component}
            inputs: {pairs: src.rows}
        """)

        migration = migrate_text(old)

        assert migration.text == expected
        assert migration.manual == []


class TestInsertedLineLayout:
    def test_counting_goes_right_after_the_last_config_line_and_comments_and_blank_lines_stay_put(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            inputs: {rows: src.rows}
            config:
              start: "[opened]"
              end: "[closed]"
              unit: day
              output: n   # the count

          # next step
          - id: out
            tool: write_csv
            config: {path: out.csv, overwrite: true, allow_empty: true}
            inputs: {rows: span.rows}
        """)
        expected = _flow("""\
          - id: span
            tool: date_diff
            inputs: {rows: src.rows}
            config:
              start: "[opened]"
              end: "[closed]"
              unit: day
              output: n   # the count
              counting: inclusive

          # next step
          - id: out
            tool: write_csv
            config: {path: out.csv, overwrite: true, allow_empty: true}
            inputs: {rows: span.rows}
        """)

        assert migrate_text(old).text == expected


class TestStripCallShapes:
    @pytest.mark.parametrize(
        "value",
        [
            '"regex_match(strip([tab]), ${rules.tabs})"',
            # Backslash escapes in a double-quoted value, before the call.
            "\"concat(regex_replace([a], '\\\\s+', ' '), strip([b]))\"",
        ],
        ids=["placeholder_argument", "escapes_before_the_call"],
    )
    def test_only_the_call_name_changes_and_every_other_byte_stays(self, value: str) -> None:
        old = _flow("""\
          - id: figured
            tool: formula
            config:
              columns:
                - name: result
                  value: VALUE
            inputs: {rows: src.rows}
        """).replace("VALUE", value)
        expected = old.replace("strip(", "trim(")
        old_value = _step_config(old, "figured")["columns"][0]["value"]

        migration = migrate_text(old)

        assert migration.text == expected
        assert _step_config(migration.text, "figured")["columns"][0]["value"] == old_value.replace("strip(", "trim(")
        assert migration.counts["strip() -> trim()"] == 1


class TestOldAndNewNameTogether:
    def test_a_config_with_both_endpoints_and_counting_is_left_alone_and_reported(self) -> None:
        old = _flow("""\
          - id: span
            tool: date_diff
            config:
              start: "[opened]"
              end: "[closed]"
              unit: day
              endpoints: both
              counting: elapsed
              output: n
            inputs: {rows: src.rows}
        """)

        migration = migrate_text(old)

        assert migration.text == old
        assert any("span" in entry for entry in migration.manual)


class TestCommandLineWithManualItems:
    """A file that still needs a person's attention is not reported as done."""

    @pytest.mark.parametrize("arguments", [[], ["--write"]], ids=["dry_run", "write"])
    def test_a_file_whose_only_issue_is_an_old_single_sheet_step_prints_it_and_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], arguments: list[str]
    ) -> None:
        path = tmp_path / "tab.yaml"
        path.write_bytes(OLD_SINGLE_SHEET.encode("utf8"))

        code = main([str(path), *arguments])
        printed = capsys.readouterr().out

        assert code == 1
        assert "tab_out" in printed
        assert path.read_bytes() == OLD_SINGLE_SHEET.encode("utf8")


class TestWhatTheFileCannotShow:
    def test_the_time_zone_setting_is_printed_as_guidance_because_only_the_data_shows_where_it_is_needed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "flow.yaml"
        path.write_text(_flow("""\
          - id: dated
            tool: cast_columns
            config:
              columns: [{column: paid_at, type: Date}]
            inputs: {rows: src.rows}
        """), encoding="utf8")

        main([str(path)])
        printed = capsys.readouterr().out

        assert "in_zone: UTC" in printed and "date_diff" in printed
        assert path.read_text(encoding="utf8").count("in_zone") == 0, "the migration cannot know, so it writes nothing"
