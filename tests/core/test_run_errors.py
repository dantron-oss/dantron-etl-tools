"""The errors a run raises, and how a caller reads them.

A step that fails raises ``StepFailedError``. A step whose expression names a missing column raises
``ColumnCheckError``, which is a kind of ``StepFailedError``, so one ``except`` catches both. Both
must survive what Python and pytest do to an exception: showing it, adding a note, copying it,
pickling it to send to another process.
"""

import copy
import pickle
import traceback
from typing import Any

import polars as pl
import pytest

from dantron_etl import (
    ColumnCheckError,
    PortRef,
    RunError,
    StepFailedError,
    load_pipeline,
    run_pipeline,
)


def _document(predicate: str = "[n] > 0") -> dict[str, Any]:
    return {
        "name": "errors",
        "steps": [
            {"id": "rows", "tool": "read_csv", "config": {"path": "unused.csv"}},
            {"id": "checked", "tool": "assert_rows", "config": {"predicate": predicate}, "inputs": {"rows": "rows.rows"}},
        ],
    }


def _failure(predicate: str = "[n] > 0") -> StepFailedError:
    spec = load_pipeline(_document(predicate))
    with pytest.raises(StepFailedError) as raised:
        run_pipeline(spec, sources={"rows.rows": pl.DataFrame({"n": [1, -1]})})
    return raised.value


def _column_failure() -> StepFailedError:
    return _failure("[missing] > 0")


class TestAStepFailureBehavesLikeAnyException:
    @pytest.mark.parametrize("make", [_failure, _column_failure], ids=["step", "column"])
    def test_a_note_and_a_traceback_can_be_attached(self, make) -> None:
        failure = make()
        failure.add_note("seen by a caller")
        failure.__traceback__ = None
        assert "seen by a caller" in "".join(traceback.format_exception(failure))

    @pytest.mark.parametrize("make", [_failure, _column_failure], ids=["step", "column"])
    def test_it_survives_a_pickle_round_trip_with_its_trace(self, make) -> None:
        failure = make()
        restored = pickle.loads(pickle.dumps(failure))
        assert type(restored) is type(failure)
        assert str(restored) == str(failure)
        assert restored.step_id == failure.step_id == "checked"
        assert restored.event == failure.event
        assert restored.trace.events == failure.trace.events
        assert restored.trace.frame(PortRef("rows", "rows")).equals(failure.trace.frame(PortRef("rows", "rows")))

    def test_it_can_be_copied(self) -> None:
        failure = _failure()
        assert copy.copy(failure).step_id == "checked"

    def test_its_fields_cannot_be_rewritten(self) -> None:
        failure = _failure()
        with pytest.raises(AttributeError):
            failure.step_id = "other"


class TestAColumnCheckIsAStepFailure:
    def test_one_except_catches_a_missing_column(self) -> None:
        failure = _column_failure()
        assert isinstance(failure, ColumnCheckError)
        assert isinstance(failure, StepFailedError)
        assert isinstance(failure, RunError)
        assert "missing" in str(failure)

    def test_a_step_failure_is_not_a_column_check(self) -> None:
        assert not isinstance(_failure(), ColumnCheckError)


class _JobLog:
    """A logger shaped like an orchestrator's: info, warning and error methods, not a logging.Logger."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, msg: str) -> None:
        self.lines.append(("info", msg))

    def warning(self, msg: str) -> None:
        self.lines.append(("warning", msg))

    def error(self, msg: str) -> None:
        self.lines.append(("error", msg))


class TestLoggingSinkTakesAnyLoggerShapedObject:
    def test_a_warning_reaches_warning_and_an_error_reaches_error(self) -> None:
        from dantron_etl.runner import logging_sink

        log = _JobLog()
        spec = load_pipeline(
            {
                "name": "log",
                "steps": [
                    {"id": "rows", "tool": "read_csv", "config": {"path": "unused.csv"}},
                    {"id": "kept", "tool": "filter", "config": {"predicate": "[n] > 5"}, "inputs": {"rows": "rows.rows"}},
                    {"id": "checked", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "kept.true"}},
                    {"id": "low", "tool": "assert_rows", "config": {"predicate": "[n] > 1"}, "inputs": {"rows": "kept.false"}},
                ],
            }
        )
        with pytest.raises(StepFailedError):
            run_pipeline(spec, sources={"rows.rows": pl.DataFrame({"n": [1, 2]})}, sink=logging_sink(log))
        assert [level for level, _ in log.lines] == ["info", "warning", "error"]
        assert log.lines[1][1].startswith("step 'kept'") and log.lines[2][1].startswith("step 'checked'")


class TestAFocusRefusalOnAFailedRunLeadsWithTheFailure:
    """A run that stops at its first step reaches no port with the focus column. The step's own
    failure is the cause, so it is the first thing a focus refusal says."""

    def _failed_at_the_first_step(self, tmp_path):
        from dantron_etl.loader import load_pipeline as load

        spec = load(
            {
                "name": "fails first",
                "steps": [
                    {"id": "entries", "tool": "read_csv", "config": {"path": str(tmp_path / "missing.csv")}},
                    {"id": "checked", "tool": "assert_rows", "config": {"predicate": "[party] != ''"}, "inputs": {"rows": "entries.rows"}},
                ],
            }
        )
        with pytest.raises(StepFailedError) as raised:
            run_pipeline(spec)
        return raised.value.trace

    def test_available_entities_names_the_failed_step_first(self, tmp_path) -> None:
        from dantron_etl import FocusSpec
        from dantron_etl.focus import MissingEvidenceError, available_entities

        trace = self._failed_at_the_first_step(tmp_path)
        with pytest.raises(MissingEvidenceError) as raised:
            available_entities(trace, FocusSpec(label="party", identity=("party",)))
        message = str(raised.value)
        assert message.startswith("The run stopped when step 'entries' failed")
        assert "missing.csv" in message
        assert "call run_pipeline" not in message, "the run already kept every frame; that advice is wrong here"

    def test_a_run_that_finished_says_nothing_about_a_failure(self) -> None:
        from dantron_etl import FocusSpec
        from dantron_etl.focus import MissingEvidenceError, available_entities

        spec = load_pipeline(_document())
        trace = run_pipeline(spec, sources={"rows.rows": pl.DataFrame({"n": [1]})})
        with pytest.raises(MissingEvidenceError) as raised:
            available_entities(trace, FocusSpec(label="party", identity=("party",)))
        assert "stopped" not in str(raised.value)
