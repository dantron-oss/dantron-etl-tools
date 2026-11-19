"""A retired name fails helpfully.

The library renamed many words a pipeline file is written in, and now accepts only the new ones.
When the loader refuses a file that still uses an old name, the refusal says so in one sentence,
gives the one command that migrates the file, and summarises the old names it found, one line per
kind of rename with a count. It stays short however big the file is. A problem that is not an old
name is still shown in full. A file refused for any other reason gets no such hint: its message is
exactly what it always was.

The expression language does the same for the one renamed function, ``strip``, now ``trim``.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from dantron_etl import REGISTRY, ConfigError, PipelineError, load_pipeline, read_pipeline
from dantron_etl.errors import RegistryError
from dantron_etl.expr import ExpressionError, compile_expr

MIGRATE_COMMAND = "python -m dantron_etl.migrate"


def _flow() -> dict[str, Any]:
    """A small valid flow in current names: read two files, pair them, count months, write out."""
    return {
        "name": "Pairs",
        "lanes": [{"id": "main", "title": "Main", "order": 1}],
        "containers": [{"id": "reading", "title": "Read both sides."}],
        "steps": [
            {"id": "orders", "tool": "read_csv", "lane": "main", "container": "reading", "config": {"path": "orders.csv"}},
            {"id": "parties", "tool": "read_csv", "lane": "main", "container": "reading", "config": {"path": "parties.csv"}},
            {
                "id": "pair_up",
                "tool": "join",
                "lane": "main",
                "config": {"left_keys": ["party_code"], "right_keys": ["code"]},
                "inputs": {"left": "orders.rows", "right": "parties.rows"},
            },
            {
                "id": "span",
                "tool": "date_diff",
                "lane": "main",
                "config": {
                    "start": "[opened]",
                    "end": "[closed]",
                    "unit": "month",
                    "counting": "inclusive",
                    "output": "months_open",
                },
                "inputs": {"rows": "pair_up.joined"},
            },
            {
                "id": "out",
                "tool": "write_csv",
                "lane": "main",
                "config": {"path": "out.csv", "overwrite": True},
                "inputs": {"rows": "span.rows"},
            },
        ],
    }


def _step(document: dict[str, Any], step_id: str) -> dict[str, Any]:
    return next(step for step in document["steps"] if step["id"] == step_id)


def _old_join_keys(document: dict[str, Any]) -> None:
    _step(document, "pair_up")["config"] = {"left_on": ["party_code"], "right_keys": ["code"]}


def _old_write_flag(document: dict[str, Any]) -> None:
    _step(document, "out")["config"] = {"path": "out.csv", "replace": True}


def _old_unit(document: dict[str, Any]) -> None:
    _step(document, "span")["config"]["unit"] = "months"


def _old_grouping_tool(document: dict[str, Any]) -> None:
    _step(document, "span")["tool"] = "connected_components"


def _old_workbook_tool(document: dict[str, Any]) -> None:
    _step(document, "out")["tool"] = "write_workbook"


def _old_top_level_groups(document: dict[str, Any]) -> None:
    document["groups"] = document.pop("containers")


def _old_step_group(document: dict[str, Any]) -> None:
    for step in document["steps"]:
        if "container" in step:
            step["group"] = step.pop("container")


def _old_lane_label(document: dict[str, Any]) -> None:
    lane = document["lanes"][0]
    lane["label"] = lane.pop("title")


def _old_join_port(document: dict[str, Any]) -> None:
    _step(document, "span")["inputs"] = {"rows": "pair_up.matched"}


# (what to make old, the error type the loader already raised, a fragment of that refusal, old name, new name)
RETIRED_CASES = [
    (_old_join_keys, ConfigError, "step 'pair_up'", "left_on", "left_keys"),
    (_old_write_flag, ConfigError, "step 'out'", "replace", "overwrite"),
    (_old_unit, ConfigError, "step 'span'", "months", "month"),
    (_old_grouping_tool, PipelineError, "no tool named 'connected_components'", "connected_components", "make_groups"),
    (_old_workbook_tool, PipelineError, "no tool named 'write_workbook'", "write_workbook", "write_excel"),
    (_old_top_level_groups, PipelineError, "unknown key 'groups'", "groups", "containers"),
    (_old_step_group, PipelineError, "unknown key 'group'", "group", "container"),
    (_old_lane_label, PipelineError, "unknown key 'label'", "label", "title"),
    (_old_join_port, PipelineError, "no output port 'matched'", "matched", "joined"),
]


def _refusal(document: dict[str, Any], expected: type[Exception]) -> str:
    with pytest.raises(expected) as refused:
        load_pipeline(document)
    assert type(refused.value) is expected
    return str(refused.value)


class TestLoaderNamesTheRename:
    @pytest.mark.parametrize(
        ("make_old", "error_type", "refusal", "old", "new"),
        RETIRED_CASES,
        ids=[case[0].__name__.removeprefix("_old_") for case in RETIRED_CASES],
    )
    def test_a_retired_name_is_refused_with_the_command_first_and_the_rename_after(
        self,
        make_old: Callable[[dict[str, Any]], None],
        error_type: type[Exception],
        refusal: str,
        old: str,
        new: str,
    ) -> None:
        document = _flow()
        make_old(document)

        message = _refusal(document, error_type)
        lines = message.splitlines()

        assert "names from before" in lines[0], message
        assert f"{MIGRATE_COMMAND} <file>" in lines[1], "the command comes straight after the first sentence"
        assert any(re.search(rf"'{old}'.*'{new}'", line) for line in lines[2:]), message
        assert refusal not in message, "a refusal caused only by an old name is not repeated"
        # Swapping the tool name alone leaves a config the new tool refuses for its own reasons, and
        # that second refusal is shown after the account of the old names. The account stays short.
        account = lines[: next((n for n, line in enumerate(lines) if line.startswith("Once renamed")), len(lines))]
        assert len(account) <= 4, message

    def test_a_strip_call_in_an_expression_is_refused_naming_trim_and_the_command(self) -> None:
        document = _flow()
        _step(document, "span")["config"]["start"] = "strip([opened])"

        message = _refusal(document, ConfigError)

        assert re.search(r"'strip'.*'trim'|strip\(\).*trim\(\)", message), message
        assert MIGRATE_COMMAND in message


def _big_old_flow(extra: int) -> dict[str, Any]:
    """The small flow plus ``extra`` writers, every step boxed with the old key and the join read as 'matched'."""
    document = _flow()
    _old_join_port(document)
    for number in range(extra):
        document["steps"].append(
            {
                "id": f"copy_{number:03d}",
                "tool": "write_csv",
                "lane": "main",
                "container": "reading",
                "config": {"path": f"copy_{number:03d}.csv"},
                "inputs": {"rows": "pair_up.matched"},
            }
        )
    _old_step_group(document)
    _old_top_level_groups(document)
    return document


class TestTheRefusalStaysShort:
    """A file of hundreds of steps gets a message a person can read, with the fix near the top."""

    def test_five_hundred_old_names_give_a_short_message_with_the_command_near_the_top(self) -> None:
        message = _refusal(_big_old_flow(500), PipelineError)
        lines = message.splitlines()

        assert len(lines) < 20, f"{len(lines)} lines"
        assert len(message) < 2_000, f"{len(message)} characters"
        assert any(MIGRATE_COMMAND in line for line in lines[:3]), message
        boxed = next(line for line in lines if "'group'" in line and "'container'" in line)
        assert re.search(r"\b502 steps\b", boxed), boxed
        assert len(re.findall(r"'copy_\d{3}'|'orders'|'parties'", boxed)) <= 3, "at most three example steps"
        read = next(line for line in lines if "'matched'" in line and "'joined'" in line)
        assert re.search(r"\b501 inputs\b", read), read

    def test_read_pipeline_names_the_file_in_the_command(self, tmp_path: Path) -> None:
        path = tmp_path / "old_flow.yaml"
        path.write_text(yaml.safe_dump(_big_old_flow(3)), encoding="utf8")

        with pytest.raises(PipelineError) as refused:
            read_pipeline(path)

        assert f"{MIGRATE_COMMAND} {path}" in str(refused.value)
        assert "<file>" not in str(refused.value)

    def test_a_mapping_has_no_path_so_the_command_says_file(self) -> None:
        message = _refusal(_big_old_flow(3), PipelineError)

        assert f"{MIGRATE_COMMAND} <file>" in message

    def test_a_problem_that_is_not_an_old_name_is_shown_beside_the_old_names(self) -> None:
        document = _big_old_flow(3)
        _step(document, "out")["wyh"] = "a misspelt key, not a renamed one"

        message = _refusal(document, PipelineError)

        assert MIGRATE_COMMAND in message
        assert re.search(r"'group'.*'container'", message), "the old names are still summarised"
        assert "step 'out'" in message and "'wyh'" in message, "the real problem is named"
        assert "did you mean 'why'" in message, "with its suggestion, as any unknown key gets"


class TestNoHintWithoutARetiredName:
    """The mirror: a refusal with no retired name in the file is exactly what it always was."""

    def test_an_unknown_tool_is_refused_exactly_as_before(self) -> None:
        document = _flow()
        _step(document, "span")["tool"] = "nope"
        with pytest.raises(RegistryError) as registry_refusal:
            REGISTRY.get("nope")

        message = _refusal(document, PipelineError)

        assert message == f"step 'span': {registry_refusal.value}"

    def test_old_words_used_only_as_values_get_no_hint(self) -> None:
        document = _flow()
        # Column names and text that happen to be old names are not retired names.
        _step(document, "pair_up")["config"] = {"left_keys": ["left_on"], "right_keys": ["matched"]}
        _step(document, "pair_up")["why"] = "Was pair_up.matched, keyed by left_on; groups and label are words."
        _step(document, "span")["tool"] = "nope"

        message = _refusal(document, PipelineError)

        assert MIGRATE_COMMAND not in message

    def test_a_config_refused_for_another_reason_gets_no_hint(self) -> None:
        document = _flow()
        _step(document, "pair_up")["config"]["how"] = "outer"

        message = _refusal(document, ConfigError)

        assert MIGRATE_COMMAND not in message

    def test_a_document_in_current_names_still_loads(self) -> None:
        pipeline = load_pipeline(_flow())

        assert pipeline.step("span").inputs["rows"].port == "joined"


class TestRenameTable:
    """The one table both the loader and the migration tool read."""

    def test_retired_names_lists_one_sentence_per_old_name_in_a_mixed_document(self) -> None:
        from dantron_etl.renames import retired_names

        document = _flow()
        _old_join_keys(document)
        _old_top_level_groups(document)

        sentences = retired_names(document)

        assert len(sentences) == 2
        about_keys = [sentence for sentence in sentences if "'left_on'" in sentence]
        assert about_keys and "'left_keys'" in about_keys[0] and "pair_up" in about_keys[0]
        assert any("'groups'" in sentence and "'containers'" in sentence for sentence in sentences)

    def test_retired_names_is_empty_for_current_names_and_old_words_used_as_values(self) -> None:
        from dantron_etl.renames import retired_names

        document = _flow()
        _step(document, "pair_up")["config"] = {"left_keys": ["left_on"], "right_keys": ["matched"]}
        _step(document, "pair_up")["why"] = "Was pair_up.matched; groups and label are only words here."

        assert retired_names(document) == []

    def test_retired_names_tells_a_join_port_from_another_tool_s_port(self) -> None:
        from dantron_etl.renames import retired_names

        document = _flow()
        _old_join_port(document)
        _step(document, "out")["inputs"] = {"rows": "span.matched"}

        sentences = retired_names(document)

        assert len(sentences) == 1
        assert "pair_up" in sentences[0] or "span" in sentences[0]
        assert "'joined'" in sentences[0]

    def test_the_command_and_the_port_map(self) -> None:
        from dantron_etl.renames import MIGRATE_COMMAND as command
        from dantron_etl.renames import port_map

        assert command == MIGRATE_COMMAND
        assert port_map() == {"tools": {"join": {"matched": "joined"}}}


class TestExpressionHint:
    def test_strip_is_refused_as_before_and_the_message_says_it_was_renamed_to_trim(self) -> None:
        with pytest.raises(ExpressionError) as refused:
            compile_expr("strip([a])")
        message = str(refused.value)

        assert "'strip' is not a function this language has" in message
        assert re.search(r"renamed to\W*trim\b", message), message

    def test_an_unknown_function_that_was_never_renamed_gets_no_rename_hint(self) -> None:
        with pytest.raises(ExpressionError) as refused:
            compile_expr("frobnicate([a])")

        assert "renamed" not in str(refused.value)

    def test_trim_compiles(self) -> None:
        compile_expr("trim([a])")
