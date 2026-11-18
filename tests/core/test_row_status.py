"""Tests for the per-row status a step reports beside an output port.

A step said *"595 of 692 values in 'column_95' did not read as Float64 and are now empty"* and the
only honest reaction to that sentence is distrust. A count names a quantity and nothing else: it
cannot say whether the 595 were figures the run destroyed or text that was never a figure, and the
difference between those two readings is the difference between a broken pipeline and a correctly
reported one. Settling it took three throwaway scripts against the source workbook. That is the
defect this closes -- **a warning that cannot be looked at** -- and the fix is not a better
sentence, it is the rows behind the sentence.

Four decisions are pinned here, and each exists because the obvious alternative fails in a specific
way.

**The status rides beside the port, never inside the frame.** A column named ``status`` would be a
column: it flows into the next join, survives a ``select_columns`` that never heard of it, turns up
in the schema comparison against the client's own engine, and lands in the published workbook. It
would then have to be carried and stripped at every one of those places, and the first place that
forgot would ship it. The status is a fact about *what a step did*, not a fact about the data, so it
travels in the envelope beside the frames -- exactly where ``notes`` already travels, and for the
same reason.

**One status row per data row, in the same order, with no cap.** The correspondence is positional
and is never re-derived afterwards, which is precisely why it is checked at the boundary on every
call rather than trusted: a status frame one row short is not a smaller answer, it is a *wrong* one,
silently attributing every warning after the gap to the wrong row. Measured on the real 672-step
run, one flag per row across 915 ports and 7,194,044 rows is 0.9 MB against 148.2 MB of port data --
under 1%. A cap would fail the feature exactly where it matters, on the frame too large to eyeball,
which is the same class of defect as a filter that quietly runs over the visible window.

**A blank is not a failure.** ``column_95`` held 97 blanks and 595 client names. The blanks did not
*fail to read* -- there was nothing there to read, which is why the existing warning's percentages
look partial and why they are right to. A refactor that collapsed the two would turn 595 into 692
and tell a reader that data was lost where none existed.

**A step that degrades nothing says nothing.** An all-fine status frame for every port of every step
is seven million rows asserting that nothing happened, and a mark that is always present is a mark
nobody reads. So an inert status is refused at the boundary, the way every other inert declaration
in this library is refused.

The message stays a count, deliberately: a sentence a person reads cannot list 595 row numbers. Only
the data behind it is complete.

Vocabulary is generic -- records, codes, amounts, periods -- because the library carries no caller's
nouns.
"""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.pipeline import PipelineSpec, PortRef, load_pipeline
from dantron_etl.preview import persist_ports
from dantron_etl.runner import StepFailedError, run_pipeline
from dantron_etl.tool import (
    STATUS_LEVELS,
    STATUS_SCHEMA,
    STATUS_VALUE_CHARS,
    ConfigField,
    PortError,
    Ports,
    ToolHelp,
    row_status,
    run_tool,
    value_caption,
)
from dantron_etl.tools.cast_columns import CastColumns
from dantron_etl.tools.select_columns import SelectColumns
from dantron_etl.tools.unpivot import Unpivot

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(levels: Sequence[str], reasons: Sequence[Sequence[Mapping[str, str]]]) -> pl.DataFrame:
    """A status frame written by hand, in the published shape, for the boundary's own tests."""
    return pl.DataFrame({"level": list(levels), "reason": [list(row) for row in reasons]}, schema=STATUS_SCHEMA)


def _levels(status: pl.DataFrame) -> list[str]:
    return status["level"].to_list()


def _reasons(status: pl.DataFrame, row: int) -> list[dict[str, str]]:
    """The reasons recorded against one row, in the order the step recorded them."""
    return status["reason"].to_list()[row]


def _columns_blamed(status: pl.DataFrame, row: int) -> list[str]:
    """Which columns a row's reasons name, in order. The whole of what 'which one' means."""
    return [entry["column"] for entry in _reasons(status, row)]


def _quoted(sentence: str) -> str:
    """The text a reason or a note quotes, read back out of it.

    A reason answers *what did this cell hold*, and the only way to check that it answered about
    this row rather than about the column, or about some fixed phrase, is to take the quoted value
    back out and compare it against the cell. Looking for the value inside the sentence instead
    would pass a reason that quoted every offending value at once.
    """
    found = re.search(r"'(.*?)'", sentence)
    assert found is not None, f"a sentence that reports a value has to quote one: {sentence!r}"
    return found.group(1)


def _exampled(note: str) -> str:
    """The offending value a note holds up as an example, which is the last thing it quotes.

    A note quotes the column first and the value last, so the value is read off the end rather than
    off the front. Read back out and compared, for the same reason a reason's is: a note holding up
    a fixed phrase, or the column's name a second time, reads perfectly and tells nobody anything.
    """
    quoted = re.findall(r"'(.*?)'", note)
    assert len(quoted) >= 2, f"a note that reports a column and an example quotes both: {note!r}"
    return quoted[-1]


def _first_detail(outputs: Ports, port: str = "rows") -> str:
    """The detail of the first reason on the first marked row of ``port``'s status."""
    status = outputs.status[port]
    for index, level in enumerate(_levels(status)):
        if level != "ok":
            return _reasons(status, index)[0]["detail"]
    raise AssertionError("this fixture is only a test if something was marked")


def _counted(note: str) -> int:
    """The first figure in a note's sentence, as an integer.

    The sentence and the rows behind it are two accounts of one fact, and the failure worth catching
    is them drifting apart -- a refactor that changed how the rows are marked while leaving the
    count computed the old way would leave a reader looking at 595 marks under a warning that says
    something else. So every case below that has both compares them rather than asserting either
    alone.
    """
    found = re.search(r"[\d,]+", note)
    assert found is not None, f"a note that reports a quantity has to carry one: {note!r}"
    return int(found.group().replace(",", ""))


def _column_95() -> list[str | None]:
    """The measured shape of the column that started this: 97 blank, 595 text, 0 figures.

    The real figures rather than a scaled-down stand-in, because the two numbers that must not be
    collapsed into one are exactly these two, and a fixture of three rows makes 'blank' and 'failed'
    easy to tell apart by eye in a way 692 rows are not.
    """
    return [None] * 97 + [f"Client {index}" for index in range(595)]


class _MarkingTool:
    """One input, two outputs, and whatever status the test hands it -- valid or not.

    Two outputs because a status is keyed by port: a tool that marks one branch and not the other is
    the ordinary case, and a single-output stub could not tell a status that reached the right port
    from one that reached the only port. It produces its ports in the reverse of its declared order
    on purpose; see the rebuild test below.
    """

    name = "marking"
    description = "Hand rows to two outputs, reporting whatever per-row status the caller supplied."
    category = "transform"
    input_ports = ("in",)
    output_ports = ("left", "right")

    def __init__(self, status: Mapping[str, pl.DataFrame] | None = None) -> None:
        self._status = status

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        return ToolHelp(detailed_description="A stub used to exercise the status channel.")

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["in"]
        return Ports({"right": frame, "left": frame}, status=self._status)


def _three_rows() -> pl.DataFrame:
    return pl.DataFrame({"record_id": [1, 2, 3], "code": ["A", "B", "C"]})


def _one_bad_row() -> pl.DataFrame:
    """A status marking the middle row of a three-row frame, which is a legal one throughout."""
    return _status(
        ["ok", "warning", "ok"],
        [[], [{"column": "code", "detail": "did not read as Float64"}], []],
    )


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


class TestTheEnvelopeCarriesAStatusBesideEachPort:
    """Where the status lives, what shape it has, and what the boundary refuses to pass on."""

    def test_a_tool_with_nothing_to_report_produces_no_status_at_all(self) -> None:
        """Twenty-eight tools exist and a handful can degrade a value; the rest must be untouched.

        Pinned against a real catalog tool run through the real boundary rather than against a stub,
        because the claim is about the shipped catalog: if adding the channel had made every tool
        report something, this is where it shows.
        """
        outputs = run_tool(
            SelectColumns(),
            Ports({"rows": _three_rows()}),
            {"columns": ["record_id"]},
        )

        assert tuple(outputs.status) == (), "a step that degraded nothing has nothing to say"
        assert tuple(Ports().status) == (), "the empty envelope defaults to nothing to say"
        assert tuple(Ports({"rows": _three_rows()}).status) == ()

    def test_the_status_rides_beside_the_port_and_survives_the_boundarys_rebuild(self) -> None:
        """Beside the frame, not inside it -- and not lost when the boundary reorders the ports.

        Two properties in one scenario because they share a failure: the frame handed downstream
        must be exactly the frame the tool produced, with no extra column for a later join to carry
        and a later `select_columns` to keep; and the boundary rebuilds the envelope to put the
        ports back in declared order, which is the step that already silently dropped `notes` once.
        A status keyed to 'left' and produced second must come back keyed to 'left'.
        """
        marked = _one_bad_row()

        outputs = run_tool(_MarkingTool({"left": marked}), Ports({"in": _three_rows()}), {})

        assert outputs.names == ("left", "right"), "the boundary hands the ports back in declared order"
        assert outputs["left"].columns == ["record_id", "code"], "no status column joined the data"
        assert outputs["left"].schema == _three_rows().schema

        assert tuple(outputs.status) == ("left",), "only the port the tool marked carries a status"
        assert _levels(outputs.status["left"]) == ["ok", "warning", "ok"]
        assert outputs.status["left"].schema == STATUS_SCHEMA

    def test_a_status_reaches_the_port_it_was_keyed_to_and_not_merely_the_first(self) -> None:
        """Keyed by port name, and the key is carried across the rebuild rather than re-derived.

        The case above marks the tool's *first* declared port, which is the one position a rebuild
        pairing statuses with ports by position rather than by name gets right by accident. A tool
        that degrades only its second branch -- the ordinary shape of a step that splits matched
        rows from unmatched ones -- would have its marks handed to the branch that came through
        clean, and every check on the way here would agree: the marks are the right height for
        either port, both ports exist, and the boundary's own inspection runs before the rebuild, so
        it inspects what the tool said rather than what the caller is handed. The mis-keying would
        first be visible to a person reading the wrong port's rows.
        """
        marked = _one_bad_row()

        outputs = run_tool(_MarkingTool({"right": marked}), Ports({"in": _three_rows()}), {})

        assert tuple(outputs.status) == ("right",), "the marked port is the one the tool named"
        assert "left" not in outputs.status, "and the clean branch stays unmarked"
        assert _levels(outputs.status["right"]) == ["ok", "warning", "ok"]

    def test_two_ports_marked_in_one_call_each_keep_their_own_marks(self) -> None:
        """Two statuses, two disagreeing level patterns, and neither may be handed to the other port.

        The sharper form of the same failure: with both ports marked, a rebuild pairing by position
        still produces two statuses of the right height under the right names, and only the contents
        are swapped -- nothing about the shape of the result is wrong. So the patterns are made to
        disagree, different marked rows blaming different columns, because that disagreement is the
        only evidence a swap leaves behind. The statuses are handed over in the reverse of the
        declared order, as this tool hands over its ports, since nothing anywhere obliges a tool to
        build its statuses in the order it declared its ports.
        """
        left = _status(
            ["warning", "ok", "ok"],
            [[{"column": "code", "detail": "did not read as Float64"}], [], []],
        )
        right = _status(
            ["ok", "ok", "warning"],
            [[], [], [{"column": "record_id", "detail": "did not read as Int64"}]],
        )

        outputs = run_tool(_MarkingTool({"right": right, "left": left}), Ports({"in": _three_rows()}), {})

        assert sorted(outputs.status) == ["left", "right"]
        assert _levels(outputs.status["left"]) == ["warning", "ok", "ok"]
        assert _levels(outputs.status["right"]) == ["ok", "ok", "warning"]
        assert _columns_blamed(outputs.status["left"], 0) == ["code"]
        assert _columns_blamed(outputs.status["right"], 2) == ["record_id"]

    def test_a_status_is_one_row_per_data_row_and_anything_else_is_refused(self) -> None:
        """The positional correspondence is the entire mechanism, so it is checked, not assumed.

        Never re-derived downstream: whatever row three of the status says is what row three of the
        frame is reported as, forever. A status one row short does not report less, it reports the
        wrong rows -- every mark after the gap slides onto its neighbour -- and that is invisible in
        every artefact built from it. Both directions are refused, because a check written against
        only the short case passes a status built by a tool that emitted a mark per *input* row
        after filtering.
        """
        short = _status(["ok", "warning"], [[], [{"column": "code", "detail": "did not read"}]])
        long = _status(
            ["ok", "warning", "ok", "ok"],
            [[], [{"column": "code", "detail": "did not read"}], [], []],
        )

        for wrong in (short, long):
            with pytest.raises(PortError) as refusal:
                run_tool(_MarkingTool({"left": wrong}), Ports({"in": _three_rows()}), {})
            assert "left" in str(refusal.value), "a refusal about a port names the port"

    def test_a_status_for_a_port_the_tool_does_not_produce_is_refused(self) -> None:
        """A status keyed to nothing is a mark nobody will ever see, and a typo that reads as silence."""
        with pytest.raises(PortError) as refusal:
            run_tool(_MarkingTool({"middle": _one_bad_row()}), Ports({"in": _three_rows()}), {})

        assert "middle" in str(refusal.value)

    def test_a_status_shaped_any_other_way_is_refused(self) -> None:
        """One published shape, or the viewer has to guess per port what it is reading.

        A level the viewer cannot draw is the same defect as a missing column: it renders as neither
        a tick nor a warning nor a cross, so the row shows nothing and reads as fine.
        """
        no_reason = pl.DataFrame({"level": ["ok", "warning", "ok"]})
        reason_as_text = pl.DataFrame(
            {"level": ["ok", "warning", "ok"], "reason": [None, "did not read as Float64", None]}
        )
        unknown_level = _status(
            ["ok", "degraded", "ok"],
            [[], [{"column": "code", "detail": "did not read"}], []],
        )

        assert "degraded" not in STATUS_LEVELS, "the fixture is only a refusal if the level is unpublished"

        for wrong in (no_reason, reason_as_text, unknown_level):
            with pytest.raises(PortError):
                run_tool(_MarkingTool({"left": wrong}), Ports({"in": _three_rows()}), {})

    def test_a_status_that_says_only_that_everything_is_fine_is_refused(self) -> None:
        """Seven million rows of 'fine' would make the feature's presence mean nothing.

        A mark that is always there is a mark nobody reads, and the point of writing a status at all
        is that its existence on a port is itself the signal. This is the same ruling as refusing a
        cast to the type a column already has: an inert declaration is refused on sight.
        """
        all_fine = _status(["ok", "ok", "ok"], [[], [], []])

        with pytest.raises(PortError):
            run_tool(_MarkingTool({"left": all_fine}), Ports({"in": _three_rows()}), {})


# ---------------------------------------------------------------------------
# Where a status is built
# ---------------------------------------------------------------------------


class TestAStatusIsCheckedWhereItIsBuiltAndNotOnlyWhereItArrives:
    """The boundary checks a finished status against its port. This is the check one level in."""

    def test_a_mark_of_a_different_height_than_the_port_is_refused_as_it_is_built(self) -> None:
        """The failure the boundary cannot see, because by then the arithmetic has already worked out.

        A caller stacking one block per melted column hands over several blocks, and the boundary
        only ever compares the total against the port: a block one row short beside a block one row
        long sums to exactly the right height, and every mark between them has slid onto a
        neighbouring row and blames a column that row did not come from. That is the precise failure
        this whole channel exists to make impossible, arriving inside the tool that builds it, and
        nothing downstream could ever detect it -- the status is well formed, the right height, and
        wrong. So each mark is measured against the height as it goes in.

        A correctly sized mark sits alongside the two bad ones, because a refusal that names every
        mark it was given sends a reader to inspect a column that is fine, and on a melt set of
        ninety columns that is the difference between a fixable message and a hunt.
        """
        marks = [
            ("q1", pl.Series([None, "did not read as Float64"])),
            ("q2", pl.Series([None, None, None])),
            ("q3", pl.Series([None, "did not read as Float64", None, None])),
        ]

        with pytest.raises(PortError) as refusal:
            row_status(marks, 3)

        message = str(refusal.value)
        assert "q1" in message and "q3" in message, "the refusal names the marks that do not fit"
        assert "q2" not in message, "and not the one that does"

        # Marks that agree with each other and disagree with the port, which is the case nothing
        # else in the stack can see: two short marks assemble into a perfectly well-formed status
        # of the wrong height, and the only fact that contradicts it is the port's own height --
        # which is precisely what is passed in here and what a check comparing the marks against
        # one another would never consult.
        agreeing = [("q1", pl.Series([None, "did not read as Float64"])), ("q2", pl.Series([None, None]))]

        with pytest.raises(PortError) as second:
            row_status(agreeing, 3)

        assert "q1" in str(second.value) and "q2" in str(second.value)

    def test_a_status_built_from_no_marks_is_a_clean_block_of_the_right_height(self) -> None:
        """What a caller stacking blocks hands in for a column that came through clean.

        Not a status worth publishing on its own -- the boundary refuses an inert one -- but it is
        how a clean column holds its place in the stack, so its height and its emptiness are both
        load-bearing: a clean block of the wrong height is the misalignment above, and a clean block
        carrying a reason would blame a column for rows it did not degrade.
        """
        clean = row_status([], 3)

        assert clean.height == 3
        assert clean.schema == STATUS_SCHEMA, "the same published shape as any other status"
        assert _levels(clean) == ["ok", "ok", "ok"]
        assert [_reasons(clean, row) for row in range(3)] == [[], [], []]


# ---------------------------------------------------------------------------
# CastColumns -- the clearest case
# ---------------------------------------------------------------------------


def _cast(frame: pl.DataFrame, *pairs: tuple[str, str], on_invalid: str | None = None) -> Ports:
    """One lenient cast through the real boundary, so the envelope is enforced on every case here."""
    config: dict[str, Any] = {"columns": [{"column": column, "type": kind} for column, kind in pairs]}
    if on_invalid is not None:
        config["on_invalid"] = on_invalid
    return run_tool(CastColumns(), Ports({"rows": frame}), config)


class TestCastColumnsMarksTheRowsItEmptied:
    """``on_invalid: empty`` already says how many. This is which."""

    def test_the_rows_that_would_not_read_are_the_rows_that_are_marked(self) -> None:
        """Position for position: a mark on row one is a claim about row one.

        Asserted as the exact set of marked positions rather than as a count of them, because a
        count matching is exactly the evidence the old warning already offered and exactly what was
        not enough.
        """
        frame = pl.DataFrame({"record_id": [1, 2, 3, 4], "amount": ["10.5", "not money", "3", "nor this"]})

        outputs = _cast(frame, ("amount", "Float64"), on_invalid="empty")
        status = outputs.status["rows"]

        assert status.height == outputs["rows"].height == 4
        assert _levels(status) == ["ok", "warning", "ok", "warning"]
        assert _columns_blamed(status, 1) == ["amount"], "a reason names the column that degraded"
        assert _columns_blamed(status, 0) == []
        assert "Float64" in _reasons(status, 1)[0]["detail"], "and says what it would not read as"

    def test_a_blank_is_not_a_failure_and_the_count_still_matches_the_marks(self) -> None:
        """97 blank and 595 failed, on the column that started this, kept apart.

        The output column cannot tell the two apart -- all 692 values are null afterwards, which is
        the whole reason the count was untrustworthy. The status can, and the number of marks it
        carries is the number the sentence reports. Collapse the two and this test says 692 where
        the run should say 595.
        """
        frame = pl.DataFrame({"record_id": list(range(692)), "column_95": _column_95()})

        outputs = _cast(frame, ("column_95", "Float64"), on_invalid="empty")
        status = outputs.status["rows"]

        assert outputs["rows"]["column_95"].null_count() == 692, "the data alone cannot tell them apart"

        assert status.height == 692
        assert _levels(status).count("warning") == 595
        assert _levels(status).count("ok") == 97
        assert _levels(status)[:97] == ["ok"] * 97, "the blanks are the first 97 rows, and they are the ok ones"

        assert len(outputs.notes) == 1
        assert _counted(outputs.notes[0]) == _levels(status).count("warning")

    def test_a_row_two_columns_degraded_keeps_both_reasons(self) -> None:
        """One row, two independent failures, and neither is allowed to overwrite the other.

        A status holding one reason per row would drop the second silently and send a reader looking
        at the wrong column -- worse than the count, which at least never claimed to say which.
        """
        frame = pl.DataFrame(
            {
                "record_id": [1, 2, 3],
                "amount": ["10.5", "not money", "3"],
                "count": ["1", "not a number", "3"],
            }
        )

        outputs = _cast(frame, ("amount", "Float64"), ("count", "Int64"), on_invalid="empty")
        status = outputs.status["rows"]

        assert _levels(status) == ["ok", "warning", "ok"]
        assert _columns_blamed(status, 1) == ["amount", "count"], "both, in the order the step cast them"
        details = {entry["column"]: entry["detail"] for entry in _reasons(status, 1)}
        assert details["amount"] != details["count"], "two reasons that read alike say one thing twice"
        assert "Float64" in details["amount"] and "Int64" in details["count"]

    def test_the_reason_quotes_the_value_that_row_actually_held(self) -> None:
        """The answer to the only question a marked row is ever asked: *what was in this cell?*

        The output column cannot answer it -- every value that failed to read is null afterwards,
        which is the whole defect -- and this is what separates a channel that says *which rows*
        from one that also says *what was lost*. It settles the user's own doubt without a script:
        read one marked row, see a client's name, and the 595 are explained.

        So the quoted text is taken back out of the reason and compared against the cell it came
        from, value for value, rather than looked for inside the sentence: a reason built from a
        fixed phrase, or from the first offending value repeated down the column, satisfies every
        weaker check and is exactly what this has to catch. Two failing rows holding different text
        is the smallest fixture in which "the same caption on every marked row" is distinguishable
        from "each row's own".
        """
        frame = pl.DataFrame({"record_id": [1, 2, 3, 4], "amount": ["10.5", "not money", "3", "nor this"]})

        outputs = _cast(frame, ("amount", "Float64"), on_invalid="empty")
        status = outputs.status["rows"]

        assert _quoted(_reasons(status, 1)[0]["detail"]) == "not money"
        assert _quoted(_reasons(status, 3)[0]["detail"]) == "nor this", "each marked row quotes its own cell"
        assert "nor this" not in _reasons(status, 1)[0]["detail"], "and not its neighbour's"

        # The one sentence a reader sees without opening anything quotes an offending value too, in
        # the same caption: "595 values failed" sends them looking, "595 values like 'Canadian
        # Business Growth Fund' failed" usually settles it where they stand.
        assert _exampled(outputs.notes[0]) == "not money"

    def test_a_long_value_is_cut_to_the_published_width_and_says_that_it_was(self) -> None:
        """A caption, not a copy -- and a reader can tell a cut value from a cell that held just that.

        Both halves matter and they fail in opposite directions. Uncapped, a column of free text
        that failed six hundred times puts a whole cell into every reason of every marked row, which
        is the frame again, in a channel whose entire justification was that it is under 1% of it.
        Cut in silence, ``'Canadian Business Growth Fund Canadian Business Growth Fu'`` cannot be
        told apart from a client whose name really is that, and a reader checking it against the
        source workbook would call it a mismatch and be wrong.

        So the width is pinned on both sides -- a value of exactly it is quoted whole, one character
        more is cut and reports its true length -- and the width itself is pinned as a number,
        because it is published: it is what a viewer lays a reason column out for, and widening it
        quietly is how a cap stops being a cap.
        """
        assert STATUS_VALUE_CHARS == 60, "the published caption width; a reason column is laid out for it"

        exact = "N" * STATUS_VALUE_CHARS
        over = "N" * (STATUS_VALUE_CHARS + 1)

        fits = _cast(pl.DataFrame({"amount": [exact]}), ("amount", "Float64"), on_invalid="empty")
        cut = _cast(pl.DataFrame({"amount": [over]}), ("amount", "Float64"), on_invalid="empty")

        quoted_whole, quoted_cut = _quoted(_first_detail(fits)), _quoted(_first_detail(cut))

        assert quoted_whole == exact, "a value that fits is quoted verbatim, with nothing said about cutting it"
        assert quoted_cut != over[:STATUS_VALUE_CHARS], "and one that does not is not quietly handed over shorter"
        assert quoted_cut.startswith(over[:STATUS_VALUE_CHARS]), "what is quoted is the front of the value"
        assert str(len(over)) in quoted_cut, "together with how long the value really was"

    def test_the_length_of_a_reason_does_not_follow_the_length_of_the_cell(self) -> None:
        """The property a cap exists for, measured rather than asserted once and assumed.

        A cap checked against a single long value passes a cap of a thousand characters, which on
        the column that prompted this would be the whole thing carried a second time. Two values
        two hundred characters apart produce reasons of *identical* length, each reporting its own
        true length -- so a marked row costs the same whatever was in the cell, which is what makes
        one flag per row across seven million rows a fraction of a percent rather than a copy of the
        data.
        """
        shorter = ("Canadian Business Growth Fund " * 7)[:200]
        longer = shorter + shorter

        first = _first_detail(_cast(pl.DataFrame({"amount": [shorter]}), ("amount", "Float64"), on_invalid="empty"))
        second = _first_detail(_cast(pl.DataFrame({"amount": [longer]}), ("amount", "Float64"), on_invalid="empty"))

        assert len(longer) == 2 * len(shorter)
        assert len(first) == len(second), "the reason is the same size whatever the cell held"
        assert shorter not in first and longer not in second, "neither carries the value it is about"
        assert str(len(shorter)) in first and str(len(longer)) in second, "each reports its own true length"

    def test_a_captioned_value_is_never_null_so_a_marked_row_is_never_unexplained(self) -> None:
        """A reason that vanished is worse than no channel: the level says degraded and says nothing more.

        :func:`polars.format` propagates a null through the whole sentence it was building, so a
        caption answering null for any input leaves a row the mask calls degraded carrying an empty
        reason -- level and reason contradicting each other inside the one artefact built to settle
        contradictions like that. Pinned as a property of the caption over a column holding every
        kind of value at once, and again over a sentence built around it, which is how it is used.

        Reached through the helper rather than through a tool, and that is the honest report rather
        than a convenience: no polars dtype answers null when cast to text -- the ones with no text
        form, a list or a duration, refuse the cast outright -- so the only value that reaches the
        guard is a null one, and ``unreadable_report``'s mask never marks a null, because a blank
        never failed to read. The guard is therefore unreachable through ``cast_columns`` or
        ``unpivot`` today. It is a guard on a published helper, and it is pinned where it lives.
        """
        frame = pl.DataFrame({"amount": ["10.5", None, "not money"]})

        captions = frame.select(value_caption("amount").alias("caption"))["caption"]
        sentences = frame.select(
            pl.format("held {}, which did not read as {}", value_caption("amount"), pl.lit("Float64"))
        ).to_series()

        assert captions.null_count() == 0, "a caption answers something about every row, including a blank one"
        assert captions[0] == "'10.5'", "a value that can be shown is shown, quoted"
        assert sentences.null_count() == 0, "so a sentence built around one never disappears"
        assert captions[1] != captions[0] != captions[2], "and the three rows do not read alike"

    def test_a_cast_that_loses_nothing_reports_no_status(self) -> None:
        """Both ways of casting cleanly, because 'always marks' and 'never marks' fail identically here.

        The lenient mode is the one at risk: it is the mode that counts, and a status written
        unconditionally inside that branch would mark every clean cast in the flow as fine.
        """
        frame = pl.DataFrame({"record_id": [1, 2, 3], "amount": ["10.5", "-3", "0"]})

        lenient = _cast(frame, ("amount", "Float64"), on_invalid="empty")
        strict = _cast(frame, ("amount", "Float64"))

        assert tuple(lenient.status) == ()
        assert tuple(strict.status) == ()
        assert lenient.notes == ()


# ---------------------------------------------------------------------------
# Unpivot -- the tool that produced the user's own warning
# ---------------------------------------------------------------------------


def _unpivot(frame: pl.DataFrame, **changes: Any) -> Ports:
    config: dict[str, Any] = {"index": ["record_id"], "into": "period", "values": "amount"}
    config.update(changes)
    return run_tool(Unpivot(), Ports({"rows": frame}), config)


class TestUnpivotMarksTheStackedRowsItEmptied:
    """The step that produced the sentence, and the one whose rows are hardest to count by hand."""

    def test_the_users_own_case_separates_the_blanks_from_the_failures(self) -> None:
        """One melted column of 692 cells: 97 that were never there, 595 that are not figures.

        This is the reported case end to end. Before it, the only account of those 692 cells was one
        sentence carrying one number; after it, every cell the sentence covers is a row somebody can
        open, and the two populations the sentence deliberately keeps apart are apart in the data
        too.
        """
        frame = pl.DataFrame({"record_id": list(range(692)), "column_95": _column_95()})

        outputs = _unpivot(frame, values_type="Float64", on_invalid="empty")
        status = outputs.status["rows"]

        assert outputs["rows"].height == 692, "one melted column, so one output row per input row"
        assert status.height == outputs["rows"].height
        assert _levels(status).count("warning") == 595
        assert _levels(status).count("ok") == 97
        assert _columns_blamed(status, 97) == ["column_95"], "the first failure names the column it came from"
        assert _reasons(status, 0) == [], "a cell that was never there did not fail to read"

        assert sum(_counted(note) for note in outputs.notes) == _levels(status).count("warning")

    def test_the_marks_follow_the_column_major_order_of_the_stacked_output(self) -> None:
        """An unpivot's output rows are not its input rows, and the status follows the output.

        All the rows of the first melted column, then all the rows of the second, in the frame's own
        column order. A status built against the *input* would be three rows where six are needed,
        and a status built in alphabetical column order would put q1's mark on a q2 row -- which
        reads perfectly and is wrong. The fixture's column order disagrees with alphabetical order
        so that the second of those is a failure rather than a coincidence.
        """
        frame = pl.DataFrame(
            {
                "record_id": [1, 2, 3],
                "q2": ["10", "not money", "30"],
                "q1": ["nor this", "2", "3"],
            }
        )

        outputs = _unpivot(frame, values_type="Float64", on_invalid="empty")
        rows, status = outputs["rows"], outputs.status["rows"]

        assert rows["period"].to_list() == ["q2", "q2", "q2", "q1", "q1", "q1"]
        assert status.height == 6
        assert _levels(status) == ["ok", "warning", "ok", "warning", "ok", "ok"]
        assert _columns_blamed(status, 1) == ["q2"]
        assert _columns_blamed(status, 3) == ["q1"], "the mark names the melted column that row came from"
        assert sum(_counted(note) for note in outputs.notes) == 2

    def test_a_clean_melted_column_holds_its_own_place_in_the_stacked_status(self) -> None:
        """A melt where one column is fine and one is not, with the fine one first.

        Every other unpivot case here melts columns that all degrade, and that is precisely the
        shape in which a clean column's *position* is never observed: if a clean block were dropped
        rather than stacked, or the blocks reordered so the degraded ones came first, the total is
        still rows times columns, so the boundary -- which compares one number against one number --
        is satisfied, and every mark lands on a row that came from the other column while naming a
        column that row has nothing to do with. The user's question was "which rows lost data" and
        the answer would be confidently wrong.

        The clean column is deliberately first. With the degraded one first, a stack that puts
        degraded blocks ahead of clean ones produces the correct answer by coincidence and the
        failure hides. The column order also disagrees with alphabetical order, so a status built by
        iterating the melt set in sorted order is a failure here rather than a coincidence too.
        """
        frame = pl.DataFrame(
            {
                "record_id": [1, 2, 3],
                "q2": ["10", "20", "30"],
                "q1": ["nor this", "2", "3"],
            }
        )

        outputs = _unpivot(frame, values_type="Float64", on_invalid="empty")
        rows, status = outputs["rows"], outputs.status["rows"]

        assert rows["period"].to_list() == ["q2", "q2", "q2", "q1", "q1", "q1"]
        assert _levels(status) == ["ok", "ok", "ok", "warning", "ok", "ok"]
        assert _columns_blamed(status, 3) == ["q1"], "the one mark is on the one row that lost a value"
        assert _reasons(status, 0) == [], "and the clean column's rows say nothing at all"

        # The property underneath those positions, stated once over the whole frame rather than
        # spot-checked: a mark is a claim about the row beside it, so the column it blames must be
        # the column that row was melted from, and that row's value must be the one that was
        # emptied. A stack in any other order satisfies the heights and breaks this on every row.
        periods, amounts = rows["period"].to_list(), rows["amount"].to_list()
        for index, level in enumerate(_levels(status)):
            blamed = _columns_blamed(status, index)
            if level == "ok":
                assert blamed == []
                continue
            assert blamed == [periods[index]], "a mark blames the column its own row came from"
            assert amounts[index] is None, "and the row it marks is one the step actually emptied"

        assert sum(_counted(note) for note in outputs.notes) == 1, "one column reported, not two"

    def test_an_unpivot_that_states_no_type_reports_no_status(self) -> None:
        """Nothing was read as anything, so nothing could fail to read.

        The melted columns agree on their own type and are stacked as they are -- the path six
        hundred of the real run's steps take, and the one that must stay free of a status frame.
        """
        frame = pl.DataFrame({"record_id": [1, 2], "q1": [1.0, 2.0], "q2": [3.0, 4.0]})

        outputs = _unpivot(frame)

        assert tuple(outputs.status) == ()
        assert outputs["rows"].height == 4


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


SETTLED = PortRef("settled", "rows")
CHECKED = PortRef("checked", "rows")


def _flow() -> PipelineSpec:
    """Three steps: a supplied source, a lenient cast that degrades, and a step that degrades nothing."""
    return load_pipeline(
        {
            "name": "a neutral fixture",
            "steps": [
                {"id": "entries", "tool": "read_csv", "config": {"path": "sources/entries.csv"}},
                {
                    "id": "settled",
                    "tool": "cast_columns",
                    "config": {
                        "columns": [{"column": "amount", "type": "Float64"}],
                        "on_invalid": "empty",
                    },
                    "inputs": {"rows": "entries.rows"},
                },
                # A check rather than a projection, because a flow whose last step merely reshapes
                # rows nobody reads is refused as a dead end. It also makes the point sharper: it
                # hands on every row exactly as it received it, and reports nothing.
                {
                    "id": "checked",
                    "tool": "assert_rows",
                    "config": {"predicate": "[record_id] > 0"},
                    "inputs": {"rows": "settled.rows"},
                },
            ],
        }
    )


def _flow_that_fails() -> PipelineSpec:
    """The same flow, with the last step's invariant made unsatisfiable so the run stops inside it.

    The same three steps deliberately: what is being observed is the trace of a run that failed
    *after* a step degraded something, so the degrading step and the failing step have to be
    different steps, and the port between them has to be one the failing step is still reading.
    """
    return load_pipeline(
        {
            "name": "a neutral fixture that stops",
            "steps": [
                {"id": "entries", "tool": "read_csv", "config": {"path": "sources/entries.csv"}},
                {
                    "id": "settled",
                    "tool": "cast_columns",
                    "config": {
                        "columns": [{"column": "amount", "type": "Float64"}],
                        "on_invalid": "empty",
                    },
                    "inputs": {"rows": "entries.rows"},
                },
                {
                    "id": "checked",
                    "tool": "assert_rows",
                    "config": {"predicate": "[record_id] > 10"},
                    "inputs": {"rows": "settled.rows"},
                },
            ],
        }
    )


def _entries() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "amount": ["10.5", "not money", "3", None],
            "code": ["A", "B", "C", "D"],
        }
    )


class TestTheTraceCarriesTheStatusPerPort:
    """A trace is the only account anybody has of a run that failed in the night; this is part of it."""

    def test_the_trace_carries_a_status_only_for_the_port_that_degraded_something(self) -> None:
        """Keyed by port, and it does not travel downstream with the rows it describes.

        'checked' hands on the very rows 'settled' emptied, and reports nothing -- correctly. A
        status is a fact about what a step did, and it did nothing to those values; propagating it
        would turn one degradation into a mark on every step after it and make the last step of a
        flow look like the culprit.
        """
        trace = run_pipeline(_flow(), sources={PortRef("entries", "rows"): _entries()}, retain="all")

        assert tuple(trace.status) == (SETTLED,)

        status = trace.status[SETTLED]
        assert status.height == trace.frame(SETTLED).height == 4
        assert _levels(status) == ["ok", "warning", "ok", "ok"], "the null was never a value that failed"
        assert _columns_blamed(status, 1) == ["amount"]
        assert CHECKED in trace.shapes, "the step that reported nothing still ran and still has a shape"

    def test_a_released_port_loses_its_status_with_its_frame(self) -> None:
        """Same lifecycle as the frame it describes, or it outlives the rows it points at.

        Under ``retain="live"`` a status kept after its frame was freed is a row reference with no
        rows -- and on the real flow it would accumulate exactly where the retention mode exists to
        stop things accumulating. The events must be identical between the two modes, which is what
        makes the status the only thing the mode changed.
        """
        sources = {PortRef("entries", "rows"): _entries()}

        whole = run_pipeline(_flow(), sources=sources, retain="all")
        live = run_pipeline(_flow(), sources=sources, retain="live")

        assert [event.notes for event in live.events] == [event.notes for event in whole.events]
        assert [event.level for event in live.events] == [event.level for event in whole.events]

        assert tuple(whole.status) == (SETTLED,)
        assert tuple(live.status) == (), "every port is released by the end of this flow, statuses with them"
        assert SETTLED in live.shapes, "and the shape stays, as it does for a released frame"

    def test_a_failed_runs_snapshot_still_accounts_for_the_rows_its_live_ports_lost(self) -> None:
        """Released with the frame is not the same as never recorded, and only a failure tells them apart.

        The case above runs a flow to the end under ``retain="live"``, where every port is freed
        before the trace is built -- so an empty status there is satisfied just as well by a run
        that never recorded one. That is the whole of the risk: a status recorded only under
        ``retain="all"`` would pass every test in this file and lose the marks exactly where they
        are worth most.

        A run that stops mid-flow is where a trace stops being a convenience and becomes the only
        account anybody has of a night's work. Its snapshot carries the frames of the ports still
        live at the moment it failed -- the emptied rows among them -- and a snapshot carrying those
        rows with no account of which of them were emptied hands the reader the exact question they
        started with. So the status must be there for a port that is, and gone for a port that is
        not: the two live and die together, which is asserted as containment over the whole
        snapshot rather than on one key, since 'kept for every port' and 'kept for the right ports'
        differ only when something has been released.
        """
        with pytest.raises(StepFailedError) as failure:
            run_pipeline(
                _flow_that_fails(),
                sources={PortRef("entries", "rows"): _entries()},
                retain="live",
            )

        snapshot = failure.value.trace
        assert failure.value.step_id == "checked"

        assert SETTLED in snapshot.frames, "the port the failing step was reading is still live"
        assert SETTLED in snapshot.status, "so the account of what it emptied is still here"
        assert _levels(snapshot.status[SETTLED]) == ["ok", "warning", "ok", "ok"]
        assert _columns_blamed(snapshot.status[SETTLED], 1) == ["amount"]
        assert snapshot.status[SETTLED].height == snapshot.frame(SETTLED).height

        assert set(snapshot.status) <= set(snapshot.frames), "no status outlives the rows it points at"
        assert PortRef("entries", "rows") not in snapshot.frames, "the source was freed once nothing read it"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestAPersistedRunKeepsTheStatusBesideThePort:
    """Written beside the port so the viewer can filter it lazily, the way row windows already are."""

    def test_the_status_is_written_beside_its_port_and_indexed_with_it(self, tmp_path: Path) -> None:
        """A separate file, its own manifest entry, and the same height as the port it describes.

        A separate file rather than extra columns in the port's own Parquet, for the same reason the
        status is not a column in the frame: the port file is what the viewer shows as the data, and
        a reader scrolling it must see what the step produced and nothing the step said about it.
        The height is asserted after a round trip through disk because that is where the positional
        correspondence would be lost -- the viewer joins the two by row position and nothing else.
        """
        frames = {"settled.rows": _entries(), "kept.rows": _entries()}
        status = {
            "settled.rows": _status(
                ["ok", "warning", "ok", "ok"], [[], [{"column": "amount", "detail": "did not read as Float64"}], [], []]
            )
        }

        manifest = persist_ports(frames, tmp_path, status=status)

        entry = manifest["settled.rows"]["status"]
        assert entry["rows"] == manifest["settled.rows"]["rows"] == 4
        assert entry["file"] != manifest["settled.rows"]["file"], "the data file stays the data alone"

        written = pl.read_parquet(tmp_path / entry["file"])
        assert written.schema == STATUS_SCHEMA
        assert written.height == pl.read_parquet(tmp_path / manifest["settled.rows"]["file"]).height
        assert _levels(written) == ["ok", "warning", "ok", "ok"]
        assert _columns_blamed(written, 1) == ["amount"], "the reason survives the round trip intact"

    def test_a_port_with_nothing_to_say_is_written_with_no_status_at_all(self, tmp_path: Path) -> None:
        """No file, no entry, and no argument required of a caller that has none.

        The absence is the signal a viewer reads to decide whether a port has a first column to draw
        at all, so an empty status file would cost 915 pointless reads per run and answer 'nothing
        here' 900 times.

        Both ways of having nothing are pinned. A caller that assembles statuses from whatever its
        run produced and finds it has none hands over an empty mapping, and 'nothing to report' must
        be a complete run rather than a refusal -- the ruling ``run_pipeline`` already takes on an
        empty cache. A signature that accepted only the omitted form would push every caller into
        writing the conditional themselves.
        """
        omitted = persist_ports({"kept.rows": _entries()}, tmp_path / "omitted")
        empty = persist_ports({"kept.rows": _entries()}, tmp_path / "empty", status={})

        assert omitted == empty, "having no status and saying so are the same run"
        assert "status" not in omitted["kept.rows"]
        assert sorted(path.name for path in (tmp_path / "empty").glob("*.parquet")) == ["kept__rows.parquet"]

    def test_the_recorded_row_count_is_read_from_the_status_and_not_from_the_port(self, tmp_path: Path) -> None:
        """The field exists to be checked against the port's own, so it cannot be a copy of it.

        The viewer joins a port and its status by row position and by nothing else, which is why the
        manifest writes both counts down: a reader loading the two lazily can tell before it draws
        anything that they do not line up. Written as the port's own height instead, the field
        agrees with the port always and by construction -- it stops being a check and becomes
        decoration, and the mismatch it was put there to reveal renders silently as marks against
        the wrong rows.

        So the two are made to disagree here, which no correct run produces -- the boundary refuses
        a mismatched status long before this, and under ``retain="live"`` a status is freed with its
        frame. That is the point: this is a writer, not a boundary. It writes down what it was
        handed, and a writer that quietly re-derived one of the two numbers it was asked to record
        would leave the run's own evidence unable to contradict itself.

        The second port is here for the contrast the manifest is read for: a port given no status
        must come back with no status, since the absence of the entry is the signal a viewer reads
        to decide whether to draw a status column at all.
        """
        frames = {"settled.rows": _entries(), "kept.rows": _entries()}
        mismatched = {
            "settled.rows": _status(
                ["ok", "warning", "ok"], [[], [{"column": "amount", "detail": "did not read as Float64"}], []]
            )
        }

        manifest = persist_ports(frames, tmp_path, status=mismatched)

        assert manifest["settled.rows"]["rows"] == 4, "the port's own height is the port's"
        assert manifest["settled.rows"]["status"]["rows"] == 3, "and the status' height is the status'"
        assert pl.read_parquet(tmp_path / manifest["settled.rows"]["status"]["file"]).height == 3

        assert "status" not in manifest["kept.rows"], "a port nothing was said about is written with nothing said"
        assert not (tmp_path / "kept__rows__status.parquet").exists()

    def test_a_status_for_a_port_that_is_not_being_written_leaves_nothing_behind(self, tmp_path: Path) -> None:
        """The ruling on an orphan: the ports being written decide what is written, and nothing else appears.

        A status keyed to a port that is not in this call's frames describes rows the reader has no
        way to open -- a status file with no port file beside it is a column of marks against a
        table that is not there, which a viewer can only draw as marks against the wrong rows or not
        at all. So it is not written, and the manifest does not mention it.

        Pinned as an invariant over the directory rather than as the absence of one name: every
        status file on disk is named by a manifest entry, and every manifest entry names a file that
        exists. That is what a reader of a persisted run relies on, and it is what a writer that
        wrote whatever it was handed would break.
        """
        frames = {"kept.rows": _entries()}
        orphaned = {"vanished.rows": _one_bad_row()}

        manifest = persist_ports(frames, tmp_path, status=orphaned)

        assert sorted(manifest) == ["kept.rows"], "the frames decide what the manifest holds"
        assert "status" not in manifest["kept.rows"]

        on_disk = {path.name for path in tmp_path.glob("*.parquet")}
        named = {entry["file"] for entry in manifest.values()} | {
            entry["status"]["file"] for entry in manifest.values() if "status" in entry
        }
        assert on_disk == named, "nothing is written that the manifest does not account for"
