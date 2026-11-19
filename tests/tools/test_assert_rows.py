"""Tests for the Assert tool: an invariant a run must satisfy, and a run that stops when it does not.

Assert takes one input port, ``rows``, states a condition every row must satisfy, and produces one
output port, also ``rows``, carrying its input unchanged. It replaces a scatter of hand-written guard
functions, and it is therefore the tool that decides whether a bad figure stops the pipeline or
reaches a report.

Five decisions are pinned here, each argued rather than assumed:

**One output port, not two.** "Which rows failed" is the first thing anyone asks, and a second port
carrying them cannot answer it: when the assertion fails the run stops, so nothing downstream ever
reads that port, and when the assertion passes the port is empty. A port that is empty whenever it is
readable and unreadable whenever it is not is a port with no possible consumer. So the offending rows
go in the message instead, and a flow that wants them as *data* rather than as a refusal reaches for
Filter, which already keeps both sides of a predicate.

**A failing assertion is a data fault.** The contract's error family covers a configuration mistake,
a port mistake, a catalog mistake and a stray failure inside a tool's own code -- and none of them
means "the data was wrong". So this tool raises :class:`DataError`, a new member of the family, and
the distinction is load bearing: a driver that retries a run on a configuration fault must not retry
one on a data fault, and ``ExecutionError`` explicitly means a failure that may have nothing to do
with the tool that raised it, decided by inspecting ``__cause__``. A deliberate refusal reported that
way would be indistinguishable from an allocation failure.

**A predicate over no rows is not satisfied.** Every row of an empty frame satisfies every condition,
so an assertion on a frame that arrived empty passes vacuously -- which is exactly the state a filter
that matched nothing leaves behind, and the most common real failure these guards exist to catch. So
an empty frame fails by default, and ``allow_empty`` is how a flow whose branch is legitimately
sometimes empty says so, in writing.

**A null result is not a pass.** A predicate is three-valued, and "could not tell" is never a
satisfied invariant. Filter offers a knob for its null rows because both dispositions are legitimate
for a *selection*; there is no knob here, because a setting that let unknown data satisfy an
assertion would exist only to silence the tool.

**The message is read later, in a log, by someone who was not there.** So it names the assertion,
says how many rows failed out of how many, and shows enough of an offending row to act on -- capped,
because an assertion that fails on forty thousand rows must not print forty thousand rows.

Vocabulary is deliberately generic -- key, amount, code -- because the library carries no caller's
nouns; these are the shapes of real guard functions, not their names.
"""

from collections.abc import Mapping
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import (
    CATEGORIES,
    ConfigError,
    DataError,
    ExecutionError,
    Ports,
    REGISTRY,
    ToolError,
    check_tool,
    run_tool,
)
from dantron_etl.config import expression_fields
from dantron_etl.tools.assert_rows import REPORTED_ROW_LIMIT, AssertRows

# ---------------------------------------------------------------------------
# Test data -- the shape a real guard function runs against
# ---------------------------------------------------------------------------


def _rows() -> pl.DataFrame:
    """One frame where every row satisfies the invariants the tests below assert."""
    return pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "amount": [10.0, 20.5, 0.5],
            "code": ["A", "B", "A"],
        }
    )


def _asserted(frame: pl.DataFrame, config: Mapping[str, Any]) -> pl.DataFrame:
    """``frame`` put through the boundary, with the single output port unwrapped."""
    result = run_tool(AssertRows(), Ports({"rows": frame}), dict(config))
    assert result.names == ("rows",), "one output port: a port carrying the failures could never be read"
    return result["rows"]


# ---------------------------------------------------------------------------
# A satisfied assertion, and a violated one
# ---------------------------------------------------------------------------


class TestAssertOutcome:
    """What passes, what stops the run, and which kind of fault a stopped run reports."""

    @pytest.mark.parametrize(
        "predicate",
        [
            pytest.param("[amount] > 0", id="every_figure_is_positive"),
            pytest.param("[code] in ('A', 'B')", id="every_code_is_one_this_flow_knows"),
            pytest.param("[key] is not null", id="every_row_is_identified"),
        ],
    )
    def test_a_satisfied_assertion_is_invisible_apart_from_its_pass_through(self, predicate: str) -> None:
        """The input arrives on the other side unchanged: same rows, same order, same dtypes.

        Invisible is the point. A step that quietly reordered or retyped what it checked would make
        every assertion in a flow a thing an author has to think about rather than add.
        """
        frame = _rows()

        result = _asserted(frame, {"predicate": predicate})

        assert_frame_equal(result, frame)

    def test_a_violated_assertion_stops_the_run_as_a_data_fault_and_not_as_any_other_kind(self) -> None:
        """The type carries the meaning, so a driver can tell the three cases apart.

        A configuration fault is worth reporting to whoever wrote the file; a stray engine failure is
        worth retrying; a data fault is worth neither and must stop the run. Reporting this as
        ``ExecutionError`` would put a deliberate refusal in the one bucket documented as possibly
        having nothing to do with the tool that raised it.
        """
        frame = _rows().with_columns(pl.Series("amount", [10.0, -1.0, 0.5]))

        with pytest.raises(DataError) as caught:
            _asserted(frame, {"predicate": "[amount] > 0"})

        assert issubclass(DataError, ToolError), "a driver catching ToolError must still catch this"
        assert not issubclass(DataError, ConfigError), "nobody's configuration was wrong"
        assert not isinstance(caught.value, ExecutionError), "this is a refusal, not a failure inside the tool"

    @pytest.mark.parametrize(
        "label",
        [pytest.param("every figure is positive", id="a_label_was_given"), pytest.param(None, id="no_label")],
    )
    def test_the_message_says_which_assertion_failed_and_how_many_rows_out_of_how_many(self, label: str | None) -> None:
        """Which assertion, the failing count, and the total -- the three facts a log reader needs.

        The count and the total together are what turn "an assertion failed" into a judgement anybody
        can make: two rows out of four thousand is a data-quality item, and four thousand out of four
        thousand is a broken step upstream.
        """
        frame = pl.DataFrame({"key": ["r1", "r2", "r3", "r4"], "amount": [10.0, -1.0, -2.0, 5.0]})
        config: dict[str, Any] = {"predicate": "[amount] > 0"}
        if label is not None:
            config["label"] = label

        with pytest.raises(DataError) as caught:
            _asserted(frame, config)

        message = str(caught.value)
        assert "[amount] > 0" in message, "the assertion has to be identifiable from the log line alone"
        assert "2" in message and "4" in message, "how many failed, out of how many"
        if label is not None:
            assert label in message, "a label is what a reader recognises the assertion by"

    def test_the_message_shows_an_offending_row_and_caps_how_much_it_prints(self) -> None:
        """Enough of a failing row to act on, and never a page of log.

        A message with no example row sends its reader back to the data to find one, which is the
        round trip the message exists to save. A message with forty thousand of them makes the log
        unreadable and buries every other line in the run.
        """
        keys = [f"r{index:05d}" for index in range(40_000)]
        amounts = [-1.0] * 39_998 + [1.0, 2.0]
        frame = pl.DataFrame({"key": keys, "amount": amounts})

        with pytest.raises(DataError) as caught:
            _asserted(frame, {"predicate": "[amount] > 0"})

        message = str(caught.value)
        assert "39998" in message and "40000" in message, "the counts are what make the scale readable"

        shown = [key for key in keys if key in message]
        assert shown, "a reader needs at least one row they can go and look at"
        assert len(shown) <= REPORTED_ROW_LIMIT, f"at most {REPORTED_ROW_LIMIT} example rows may be printed"
        assert len(message) < 2_000, "a refusal is one log line's worth, whatever the size of the failure"

    def test_a_null_predicate_result_is_not_silently_a_pass(self) -> None:
        """A row nobody could judge is not a row that passed, and it is reported as a failure.

        This is the one place the tool deliberately differs from a filter, which routes its null rows
        to ``false`` and offers a knob to move them. An assertion has no such knob: a missing figure
        is exactly the state a guard is placed to catch, and a null that passed would make the guard
        strongest on clean data and silent on dirty.
        """
        frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.0, None]})

        with pytest.raises(DataError) as caught:
            _asserted(frame, {"predicate": "[amount] > 0"})

        message = str(caught.value)
        assert "r2" in message, "the row that could not be judged is the row to look at"
        assert "1" in message and "2" in message, "one row of two failed"

    @pytest.mark.parametrize(
        ("config", "expectation"),
        [
            pytest.param({"predicate": "[amount] > 0"}, "refused", id="an_empty_frame_fails_by_default"),
            pytest.param({"predicate": "[amount] > 0", "allow_empty": True}, "passed", id="unless_it_is_allowed_to"),
        ],
    )
    def test_a_frame_with_no_rows_fails_unless_emptiness_is_allowed(
        self,
        config: Mapping[str, Any],
        expectation: str,
    ) -> None:
        """Every row of an empty frame satisfies every condition, which is why this needs a ruling.

        A filter that matched nothing leaves an empty frame behind, every figure built from it is
        zero, and every assertion on it passes vacuously -- the exact silence these guards replace.
        So an empty frame is a failure unless a flow says in writing that this branch may be empty.
        """
        frame = _rows().head(0)

        if expectation == "passed":
            result = _asserted(frame, config)
            assert result.height == 0
            assert result.schema == frame.schema, "an allowed empty frame still passes its schema on"
            return

        with pytest.raises(DataError) as caught:
            _asserted(frame, config)

        message = str(caught.value).lower()
        assert "empty" in message or "no rows" in message, "the refusal must say the frame was empty"


# ---------------------------------------------------------------------------
# A configuration fault is not a data fault
# ---------------------------------------------------------------------------


class TestAssertRefusals:
    """The two kinds of wrong this tool can meet, and the two different types it reports them as."""

    @pytest.mark.parametrize(
        ("predicate", "named"),
        [
            pytest.param("[amount]", "amount", id="a_condition_that_is_not_one"),
            pytest.param("[amoun] > 0", "amoun", id="a_column_that_is_not_there"),
            pytest.param("[code] = '${rules.wanted}'", "${rules.wanted}", id="an_unresolved_placeholder"),
        ],
    )
    def test_a_predicate_the_schema_refuses_is_a_configuration_fault_not_a_data_fault(
        self,
        predicate: str,
        named: str,
    ) -> None:
        """An assertion nobody could evaluate says nothing about the data, and must not claim to.

        Reported as a data fault, each of these would send a reader to look for a bad figure that is
        not there -- and the placeholder case is the one that matters most, because a predicate
        carrying an unresolved placeholder would otherwise compile as ordinary text, fail on every
        row, and be reported as forty thousand bad rows.
        """
        with pytest.raises(ConfigError) as caught:
            _asserted(_rows(), {"predicate": predicate})

        assert named in str(caught.value)
        assert not isinstance(caught.value, DataError), "the configuration was wrong, not the data"

    def test_assert_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the data must be unchanged too."""
        frame = _rows()
        before = frame.clone()

        _asserted(frame, {"predicate": "[amount] > 0"})

        assert_frame_equal(frame, before)

    def test_the_template_states_one_invariant_and_says_nothing_about_row_counts(self) -> None:
        """A predicate, a label, and the empty-frame ruling -- and deliberately no count identity.

        "This frame has as many rows as it had before" is the other shape the hand-written guards
        use, and it belongs to a tool with two inputs rather than to this one: it compares two frames,
        and this tool can only see the frame in front of it. Folding it in would give Assert a second,
        mutually exclusive mode, which is two tools sharing a name.
        """
        template = AssertRows().config_template()

        assert set(template) == {"predicate", "label", "allow_empty"}
        assert template["predicate"].required is True
        assert expression_fields(template) == ("predicate",), "the loader seam is the declared type"
        assert template["label"].required is False
        assert template["allow_empty"].type == "boolean"
        assert template["allow_empty"].default is False, "an empty frame fails until a flow says otherwise"


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestAssertInTheCatalog:
    """Assert is described, documented with examples that actually run, and resolvable by name."""

    def test_assert_conforms_and_is_documented_as_the_tool_that_stops_a_run(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output.

        Every example a catalog ships has to *run*, so no example can document a violated assertion:
        the failure story lives in the notes, and the examples show the pass-through and the empty
        frame ruling instead.
        """
        tool = AssertRows()

        check_tool(tool)

        assert tool.name == "assert_rows", "'assert' is a Python keyword, so the module cannot carry that name"
        assert tool.category == "assert"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        help_ = tool.help()
        assert len(help_.examples) >= 2, "one satisfied assertion, and one showing the empty-frame ruling"
        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "null" in documented, "that a null result is not a pass cannot be guessed from the config"
        assert "empty" in documented, "nor can the ruling on a frame that arrived empty"

    def test_assert_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "assert_rows" in REGISTRY
        assert isinstance(REGISTRY.get("assert_rows"), AssertRows)
