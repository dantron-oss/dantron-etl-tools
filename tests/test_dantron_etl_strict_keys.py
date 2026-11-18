"""The loader refuses a key it does not know, at every level of a pipeline file.

A misspelt config field was already refused. A misspelt key around it was not: ``step:`` for
``steps:`` loaded a flow with no steps, and ``gruop:`` left a step outside its box. Both loaded
cleanly and said nothing. These tests pin the refusal, its wording, and the mirror case: a file that
uses every allowed key still loads and keeps what it said.
"""

from typing import Any

import pytest

from dantron_etl.pipeline import PipelineError, load_pipeline


def _document(**changes: Any) -> dict[str, Any]:
    """A small valid flow: read a file, write it out. Every allowed key is used once somewhere."""
    document: dict[str, Any] = {
        "name": "Copy",
        "lanes": [{"id": "a", "label": "A", "order": 1}],
        "groups": [{"id": "move", "title": "Read the file and write it out again."}],
        "steps": [
            {
                "id": "src",
                "tool": "read_csv",
                "lane": "a",
                "stage": "read",
                "group": "move",
                "why": "The file as it arrived.",
                "config": {"path": "in.csv"},
            },
            {
                "id": "out",
                "tool": "write_csv",
                "lane": "a",
                "stage": "write",
                "group": "move",
                "why": "The same rows, written out.",
                "config": {"path": "out.csv"},
                "inputs": {"rows": "src.rows"},
            },
        ],
    }
    document.update(changes)
    return document


def _refusal(document: dict[str, Any]) -> str:
    with pytest.raises(PipelineError) as refused:
        load_pipeline(document)
    return str(refused.value)


def test_a_file_using_every_allowed_key_loads_and_keeps_each_one() -> None:
    spec = load_pipeline(_document())

    written = spec.step("out")
    assert (written.lane, written.stage, written.group, written.why) == ("a", "write", "move", "The same rows, written out.")
    assert [lane.label for lane in spec.lanes] == ["A"]
    assert [group.title for group in spec.groups] == ["Read the file and write it out again."]


def test_a_misspelt_top_level_key_is_refused_with_the_key_it_was_meant_to_be() -> None:
    document = _document()
    document["step"] = document.pop("steps")

    message = _refusal(document)

    assert "'step'" in message
    assert "did you mean 'steps'" in message
    for allowed in ("name", "steps", "lanes", "groups"):
        assert allowed in message


@pytest.mark.parametrize(("typo", "meant"), [("gruop", "group"), ("wy", "why"), ("inptus", "inputs")])
def test_a_misspelt_step_key_is_refused_naming_the_step_and_the_key_it_was_meant_to_be(typo: str, meant: str) -> None:
    document = _document()
    step = document["steps"][1]
    step[typo] = step.pop(meant)

    message = _refusal(document)

    assert "step 'out'" in message
    assert repr(typo) in message
    assert f"did you mean {meant!r}" in message


def test_an_unknown_step_key_unlike_any_allowed_one_lists_what_is_allowed_and_guesses_nothing() -> None:
    document = _document()
    document["steps"][0]["colour"] = "blue"

    message = _refusal(document)

    assert "step 'src'" in message
    assert "'colour'" in message
    assert "did you mean" not in message
    for allowed in ("id", "tool", "config", "inputs", "lane", "stage", "group", "why"):
        assert allowed in message


def test_a_misspelt_lane_key_is_refused_naming_the_lane() -> None:
    document = _document()
    lane = document["lanes"][0]
    lane["lable"] = lane.pop("label")

    message = _refusal(document)

    assert "lane 'a'" in message
    assert "did you mean 'label'" in message


def test_a_misspelt_group_key_is_refused_naming_the_group() -> None:
    document = _document()
    group = document["groups"][0]
    group["titel"] = group.pop("title")

    message = _refusal(document)

    assert "group 'move'" in message
    assert "did you mean 'title'" in message


@pytest.mark.parametrize(
    ("where", "missing"),
    [("document", "name"), ("document", "steps"), ("step", "tool"), ("step", "id"), ("lane", "order")],
)
def test_a_missing_required_key_is_refused_by_name_and_not_as_a_bare_key_error(where: str, missing: str) -> None:
    document = _document()
    holder = {"document": document, "step": document["steps"][0], "lane": document["lanes"][0]}[where]
    del holder[missing]

    message = _refusal(document)

    assert repr(missing) in message


def test_a_flow_with_no_steps_is_refused_rather_than_run_as_an_empty_success() -> None:
    message = _refusal({"name": "Empty", "steps": []})

    assert "no steps" in message


@pytest.mark.parametrize("kind", ["lanes", "groups"])
def test_two_lanes_or_two_groups_sharing_an_id_are_refused(kind: str) -> None:
    document = _document()
    first = document[kind][0]
    document[kind].append({**first, "order": 2} if kind == "lanes" else {**first, "title": "A second title."})

    message = _refusal(document)

    assert "duplicate" in message
    assert repr(first["id"]) in message


@pytest.mark.parametrize("document", [None, ["steps"], "steps: []"])
def test_a_file_that_is_not_a_mapping_is_refused_by_what_it_holds(document: Any) -> None:
    message = _refusal(document)

    assert type(document).__name__ in message
