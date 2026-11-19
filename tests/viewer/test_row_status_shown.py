"""Tests for a run's per-row marks reaching the reader: written down, served, and drawn first.

Cycle one built the mechanism and stopped one step short of anybody seeing it. A step still reports
*"595 of 692 values in 'column_95' did not read as Float64 and are now empty"*, the rows behind that
sentence now exist as a status frame beside the port -- and a reader opening the run is shown the
same count over the same table it always was. Two things are missing between the two, and this file
pins both.

**Nothing writes the marks down.** ``persist_ports`` takes a ``status=`` and no caller passes one,
so a run directory is written from a trace that knows exactly which rows were emptied and keeps
none of it. A writer that takes the frames out of a trace and leaves the marks in it is the whole
defect: both are on the same object, one call apart, and the difference is invisible until a reader
asks a question the directory cannot answer. So a run is written *from the trace* -- one call, both
halves -- and what is pinned here is the property, not the call: a directory written from a run that
degraded something carries the marks, and one written from a run that degraded nothing carries
none.

**The viewer cannot show them.** What the reader asked for is Alteryx's answer: a first column on
every output, a tick or a warning or a cross per row, and a filter that narrows to the flagged rows
so the same figures can be read in and out until the warning is understood and accepted. That is
four separate obligations and each has an obvious wrong version:

*The marks belong to the window, not to the port.* The table reads a window at a time
(``api/rows?port=...&offset=...&limit=...``), so the marks a window carries must be that window's own
slice. A status column filled from the first N marks lines up perfectly on the first page, is wrong
on every page after it, and is wrong in the way that matters most -- it attributes a warning to a
row that never had one. Every alignment case below therefore reads a *later* window, and checks the
mark against what that row's own value was, rather than against a count.

*A column of ticks trains a reader to ignore the column.* Almost every port in a real flow degraded
nothing, so a port with no marks draws no status column at all: its table, its search boxes, its
lazy body and its fetch are exactly what they were.

*A filter on a level is not a search for a word.* The per-column boxes are substring searches, which
is right for a value a person half-remembers and wrong for a vocabulary of three. ``warn`` would
match ``warning`` and ``error`` would match a client's own column holding the word; both read as a
filter that works. So the level travels as its own parameter, is checked against the levels the
library publishes, and is refused when it is not one of them.

*A glyph is not a sentence.* A tick, a warning mark and a cross tell a reader looking at the screen
which rows to open and tell a screen reader nothing, and a reader who cannot get from the mark back
to *"held 'Canadian Business Growth Fund', which did not read as Float64"* is back to distrusting a
count. So the level reaches the markup as something other than a character, and the reason -- the
column that degraded, and what the cell held -- is reachable from the flagged row itself.

The shape being pinned, so an implementation has something to aim at:

    preview.persist_trace(trace, directory)     the run's ports AND its marks, written together
    Window.status                               one entry per row of THIS window, or None
                                                {"level": "warning",
                                                 "reason": [{"column": ..., "detail": ...}]}
    PortStore.window(..., level="warning")      narrows to a level, refuses anything else
    GET api/rows?...&level=warning              the same, over the wire, with "status" in the payload

    <th class="lvl">          first in the header row, carrying no dtype, ahead of the data
    <th class="lvl"><select>  the level control, offering exactly the published levels
    <input class="flt" data-c="1" ...>          the boxes still name the cell they narrow

Vocabulary is generic -- records, parties, amounts, codes -- because the library carries no caller's
nouns.
"""

import json
import re
import shutil
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.focus import FocusSpec
from dantron_etl import Pipeline, PortRef, load_pipeline
from dantron_etl.preview import FILTER_PREFIX, MANIFEST, PortStore, PreviewError, serve
from dantron_etl import PipelineTrace, run_pipeline
from dantron_etl.authoring import STATUS_LEVELS

# The class that says "this cell is the step talking about the row, not the row". One name, worn by
# the header, by the control under it and by the cell the script draws, because the three are one
# column: a stylesheet that had to name three things would be free to style two of them.
MARK_CLASS = "lvl"

SETTLED = PortRef("settled", "rows")
ENTRIES = PortRef("entries", "rows")

FOCUS = FocusSpec(label="party", identity=("party",))


# ---------------------------------------------------------------------------
# A flow that degrades something, and one that degrades nothing
# ---------------------------------------------------------------------------


def _entries() -> pl.DataFrame:
    """Twelve rows, with two different columns failing on two different sets of them.

    Twelve rather than three because every alignment case below reads a window that starts part way
    down the port, and a fixture smaller than two windows cannot tell a mark that was sliced from a
    mark that was taken off the top.

    Every offending value is distinct and carries its own row's number, so a reason can be read back
    out and matched to the row it was recorded against. A fixture where every bad value was the same
    text would pass just as happily with every mark on the wrong row.
    """
    return pl.DataFrame(
        {
            "record_id": list(range(1, 13)),
            "party": ["Northwind Supply", "Borealis Freight", "Cirrus Analytics"] * 4,
            "amount": [f"{i * 10}.5" if i % 2 == 0 else f"not money {i}" for i in range(12)],
            "code": [str(i) if i % 3 else f"code {i}" for i in range(12)],
        }
    )


def _clean_entries() -> pl.DataFrame:
    """The same twelve rows with nothing in them that fails to read, for the negative control."""
    return _entries().with_columns(
        pl.Series("amount", [f"{i * 10}.5" for i in range(12)]),
        pl.Series("code", [str(i) for i in range(12)]),
    )


def _flow() -> Pipeline:
    """Four steps: a source, two casts that empty different rows, and a check that touches nothing.

    Two degrading steps rather than one, because a writer that stacked every mark it found under
    whichever port it reached first would be invisible against a run with one. They degrade
    different columns on different rows for the same reason.

    The last step hands its rows straight on and reports nothing, which is the port this file uses
    for every "a port nothing was said about" case: a derived port with no marks of its own is the
    ordinary shape of a flow, and it is the one that must be left exactly as it was.
    """
    return load_pipeline(
        {
            "name": "a neutral fixture that degrades something",
            "lanes": [
                {"id": "source", "title": "Source", "order": 1},
                {"id": "shaping", "title": "Shaping", "order": 2},
            ],
            "steps": [
                {"id": "entries", "tool": "read_csv", "lane": "source", "config": {"path": "sources/entries.csv"}},
                {
                    "id": "settled",
                    "tool": "cast_columns",
                    "lane": "shaping",
                    "config": {"columns": [{"column": "amount", "type": "Float64"}], "on_invalid": "set_null"},
                    "inputs": {"rows": "entries.rows"},
                },
                {
                    "id": "graded",
                    "tool": "cast_columns",
                    "lane": "shaping",
                    "config": {"columns": [{"column": "code", "type": "Int64"}], "on_invalid": "set_null"},
                    "inputs": {"rows": "settled.rows"},
                },
                {
                    "id": "checked",
                    "tool": "assert_rows",
                    "lane": "shaping",
                    "config": {"predicate": "[record_id] > 0"},
                    "inputs": {"rows": "graded.rows"},
                },
            ],
        }
    )


def _run(frame: pl.DataFrame | None = None) -> PipelineTrace:
    return run_pipeline(_flow(), sources={ENTRIES: _entries() if frame is None else frame}, retain="all")


def _write(trace: PipelineTrace, directory: Path) -> Path:
    """The run, written down: both the ports and what the steps said about their rows.

    The writer is reached for here rather than imported at the top of the file, in the shape the
    existing page tests use for the same reason: a suite run before it exists fails on the cases
    that are about writing a run, rather than collecting nothing at all and reporting one error
    about an import where fifteen statements about the viewer should be.
    """
    from dantron_etl.preview import persist_trace

    persist_trace(trace, directory)
    return directory


@pytest.fixture(scope="module")
def trace() -> PipelineTrace:
    """One run of the degrading flow. Pure, so every case below reads the same evidence."""
    return _run()


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory, trace: PipelineTrace) -> Path:
    """That run, written down as a run directory -- which is the only thing the viewer ever reads."""
    directory = tmp_path_factory.mktemp("run")
    _write(trace, directory)
    return directory


@pytest.fixture(scope="module")
def store(written: Path) -> PortStore:
    return PortStore(written)


@pytest.fixture(scope="module")
def page(trace: PipelineTrace) -> str:
    from dantron_etl.viewer import build

    return build(_flow(), trace, FOCUS)


@pytest.fixture(scope="module")
def panels(trace: PipelineTrace) -> dict[str, str]:
    """Every step's data panel, as the service hands them over, for the flow that degraded something.

    Imported inside the fixture, in the shape the existing page tests use: a suite run before any of
    this exists fails on the cases that are about it rather than collecting nothing at all.
    """
    from dantron_etl.viewer import build_panels

    return build_panels(_flow(), trace, FOCUS)


# ---------------------------------------------------------------------------
# Reading what was written
# ---------------------------------------------------------------------------


def _held(column: str) -> dict[str, str]:
    """What the source held in ``column``, keyed by the record id, as text.

    The point of the channel: after the step the cell is null, so the only way to check that a mark
    landed on the right row is to compare its reason against what that row held *before*.
    """
    frame = _entries()
    return {str(record): value for record, value in zip(frame["record_id"], frame[column], strict=True)}


def _quoted(detail: str) -> str:
    """The value a reason quotes, read back out of the sentence rather than searched for inside it.

    Asking whether the value appears in the reason would pass a reason that quoted every offending
    value at once, which is exactly the shape a per-column message takes when it is copied onto
    every row.
    """
    found = re.search(r"'(.*?)'", detail)
    assert found is not None, f"a reason that reports a value has to quote one: {detail!r}"
    return found.group(1)


def _levels(window: Any) -> list[str]:
    assert window.status is not None, "this window carries no marks at all, so there is nothing to read"
    return [mark["level"] for mark in window.status]


def _column_of(row: Any, window: Any, name: str) -> str | None:
    return row[window.columns.index(name)]


def _status_files(directory: Path) -> set[str]:
    """Every file in a run directory that is a port's marks rather than a port's rows."""
    ports = {entry["file"] for entry in json.loads((directory / MANIFEST).read_text()).values()}
    return {path.name for path in directory.glob("*.parquet")} - ports


# ---------------------------------------------------------------------------
# A run directory carries what the run said
# ---------------------------------------------------------------------------


class TestARunIsWrittenWithTheMarksItsStepsMade:
    """The half nobody wrote: a trace that knows which rows were emptied, written down whole."""

    def test_a_run_written_from_a_trace_carries_the_marks_of_every_port_that_degraded_something(
        self, store: PortStore
    ) -> None:
        """Both halves out of one trace, keyed the way the page and the query string spell a port.

        Read back through the store rather than off the manifest, because the manifest is the index
        and the store is what a reader's request goes through: a run whose marks are on disk and
        unreachable by name is a run with no marks in it.

        Asserted over the whole run rather than on one port, and on both sides: the ports that
        degraded something have marks, and the ports that did not have none. A writer that wrote a
        status for every port would satisfy the first half and make the column meaningless.
        """
        marked = {port for port in store.ports() if store.window(port, limit=1).status is not None}

        assert marked == {"settled.rows", "graded.rows"}, (
            "the run directory does not carry the marks of exactly the ports whose steps emptied something"
        )
        assert set(store.ports()) == {"entries.rows", "settled.rows", "graded.rows", "checked.rows"}, (
            "the ports themselves are unchanged by carrying marks beside some of them"
        )

    def test_a_run_whose_steps_degraded_nothing_is_written_with_no_marks_at_all(
        self, tmp_path: Path, written: Path
    ) -> None:
        """The same flow over values that all read, which is the only honest negative control.

        Same steps, same tools, same ports -- the one difference is that nothing failed to parse. A
        run written from it holds no marks anywhere on disk, because the absence of a status is what
        the viewer reads to decide whether the port has a column to draw: a file per port saying
        that every row is fine costs a read apiece to be told nothing, and puts a column of ticks in
        front of a reader on every table in the flow.

        The directory that did degrade something is measured beside it, so this cannot pass by the
        writer having written no marks for anybody.
        """
        clean = tmp_path / "clean"
        _write(_run(_clean_entries()), clean)

        assert _status_files(clean) == set(), "a run that degraded nothing left marks behind anyway"
        assert _status_files(written), "the run that did degrade something left none either, so this proves nothing"

        store = PortStore(clean)
        assert set(store.ports()) == {"entries.rows", "settled.rows", "graded.rows", "checked.rows"}
        for port in store.ports():
            assert store.window(port, limit=1).status is None, f"{port} reports marks nothing made"

    def test_each_ports_marks_are_indexed_against_that_port_and_not_against_its_neighbour(
        self, store: PortStore
    ) -> None:
        """Two degrading steps, two sets of marks, and nothing to stop a writer swapping them.

        The two ports are the same height and the same shape, so a writer that paired the frames and
        the statuses by iteration order rather than by port would produce a directory that loads,
        serves and draws -- with every warning on the wrong rows, blaming a column the step never
        touched. What tells them apart is which column each reason names and which rows carry one.
        """
        settled = store.window("settled.rows", limit=12)
        graded = store.window("graded.rows", limit=12)

        assert {reason["column"] for mark in settled.status for reason in mark["reason"]} == {"amount"}
        assert {reason["column"] for mark in graded.status for reason in mark["reason"]} == {"code"}

        # Different rows, too: the columns alone would still be satisfied by two statuses that had
        # been written to the right ports and were internally identical.
        assert [i for i, level in enumerate(_levels(settled)) if level != "ok"] != [
            i for i, level in enumerate(_levels(graded)) if level != "ok"
        ]


# ---------------------------------------------------------------------------
# A window carries its own rows' marks
# ---------------------------------------------------------------------------


class TestAWindowCarriesTheMarksOfTheRowsItHandedBack:
    """Positional, one for one, over whatever slice of the port the reader is looking at."""

    def test_a_window_carries_one_mark_per_row_it_hands_back(self, store: PortStore) -> None:
        """The baseline: as many marks as rows, each naming the column that emptied its cell.

        The level is checked against what the row held rather than against a fixed list, so this
        stays a statement about the data and not about a fixture: a value that read as a figure is
        ``ok``, one that did not is flagged, and a blank -- there are none here, and there were 97
        in the column that started all this -- is never a failure.
        """
        window = store.window("settled.rows", limit=12)

        assert window.status is not None, "the port whose step emptied six of its rows reports nothing"
        assert len(window.status) == len(window.rows) == 12

        held = _held("amount")
        for row, mark in zip(window.rows, window.status, strict=True):
            record = _column_of(row, window, "record_id")
            readable = _is_a_number(held[record])
            assert (mark["level"] == "ok") == readable, f"record {record} held {held[record]!r} and is {mark['level']}"
            if not readable:
                assert [reason["column"] for reason in mark["reason"]] == ["amount"]
                assert _quoted(mark["reason"][0]["detail"]) == held[record]

    def test_a_later_window_carries_the_marks_of_the_rows_it_returned_and_not_of_the_first_ones(
        self, store: PortStore
    ) -> None:
        """The defect this whole channel would otherwise re-introduce, one layer up.

        The correspondence between a row and its mark is positional and is never re-derived, so a
        window that took the first four marks for the rows at offset eight would draw a status
        column that is correct on page one and wrong on every page after it -- not wrong by being
        empty, wrong by putting a warning on a row that never had one and a tick on a row that did.

        Checked by reading the reason back out and comparing it to what *that* row held, which is
        the only comparison a mark taken from the wrong place cannot survive. Both windows are read
        so the case cannot pass by the port having the same marks all the way down.
        """
        first = store.window("settled.rows", offset=0, limit=4)
        later = store.window("settled.rows", offset=8, limit=4)

        assert _levels(later) != _levels(first) or [r[0] for r in later.rows] != [r[0] for r in first.rows], (
            "the two windows are indistinguishable, so this fixture cannot tell a slice from a repeat"
        )

        held = _held("amount")
        assert later.offset == 8
        assert [_column_of(row, later, "record_id") for row in later.rows] == ["9", "10", "11", "12"]
        for row, mark in zip(later.rows, later.status, strict=True):
            record = _column_of(row, later, "record_id")
            if mark["level"] == "ok":
                continue
            assert _quoted(mark["reason"][0]["detail"]) == held[record], (
                f"the mark drawn against record {record} quotes {mark['reason'][0]['detail']!r}, which is another "
                "row's value: the window's marks were taken from the top of the port rather than from the window"
            )

    def test_a_port_nothing_was_said_about_reports_no_marks_rather_than_a_column_of_ticks(
        self, store: PortStore
    ) -> None:
        """Absent, not all-fine. The two are one word apart and they draw completely differently.

        A port reporting twelve ``ok``s is a port with a status column, and a column that is always
        a tick is a column a reader stops looking at -- which costs the flagged ports the attention
        the whole feature exists to buy. So a port with nothing said about it answers ``None``, and
        the rows it hands back are exactly what they were.
        """
        window = store.window("checked.rows", limit=12)

        assert window.status is None
        assert len(window.rows) == 12, "the rows of an unmarked port are unchanged"
        assert window.total == window.matched == 12

    def test_narrowing_to_a_level_answers_with_those_rows_and_counts_only_them(self, store: PortStore) -> None:
        """The reader's own workflow: show me the rows this warning is about, and only those.

        The count moves with the filter -- ``matched`` is what was narrowed to, ``total`` is still
        the port -- because the note under the table reads "1-2 of 6 matching (of 12)", and a filter
        that narrowed the rows while leaving the count at the port's height would tell a reader that
        half the flagged rows are missing.

        Read at an offset as well as from the top, because slicing after narrowing and narrowing
        after slicing agree on the first window and disagree on every one after it: the second
        answers with however many of the first ``limit`` rows happened to be flagged.
        """
        flagged = store.window("settled.rows", limit=12, level="warning")

        assert flagged.total == 12, "the port's own height is what the port holds, filter or no filter"
        assert flagged.matched == 6
        assert _levels(flagged) == ["warning"] * 6
        assert len(flagged.rows) == 6

        held = _held("amount")
        deeper = store.window("settled.rows", offset=4, limit=2, level="warning")
        assert deeper.matched == 6
        assert [_column_of(row, deeper, "record_id") for row in deeper.rows] == ["10", "12"]
        for row, mark in zip(deeper.rows, deeper.status, strict=True):
            assert _quoted(mark["reason"][0]["detail"]) == held[_column_of(row, deeper, "record_id")]

    def test_a_level_and_a_column_filter_narrow_together_rather_than_one_replacing_the_other(
        self, store: PortStore
    ) -> None:
        """Both are narrowings of the same port, and a reader applies them at once by typing.

        The reader's question is "which of *this party's* rows is the warning about", and the two
        controls sit in the same row of the same table. An implementation that ran the level filter
        over the window the column filter had already cut -- or the other way about -- would answer
        with a subset that looks entirely plausible and is not the intersection.

        Pinned against both halves separately, so the intersection cannot be satisfied by either
        filter having quietly done nothing.
        """
        party = {"party": "Cirrus Analytics"}

        both = store.window("settled.rows", limit=12, level="warning", filters=party)
        one_party = store.window("settled.rows", limit=12, filters=party)
        every_warning = store.window("settled.rows", limit=12, level="warning")

        assert one_party.matched == 4
        assert every_warning.matched == 6
        assert both.matched == 2, "the two narrowings are not being applied to the same rows"
        assert _levels(both) == ["warning", "warning"]
        assert {_column_of(row, both, "party") for row in both.rows} == {"Cirrus Analytics"}
        assert [_column_of(row, both, "record_id") for row in both.rows] == ["6", "12"]


def _restated(written: Path, directory: Path, port: str, by: int) -> Path:
    """A copy of the run with one port's marks ``by`` rows taller (or, negative, shorter) than it.

    The shape of a directory that was assembled by hand, or truncated after it was written, or
    written by a caller that built its statuses from one run and its frames from another. Both
    halves are moved together -- the parquet and the count the manifest wrote down beside it --
    because a tamper that moved only one would be a test of bookkeeping rather than of the pair.
    """
    shutil.copytree(written, directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    beside = manifest[port]["status"]
    marks = pl.read_parquet(directory / beside["file"])
    changed = pl.concat([marks, marks.head(by)]) if by > 0 else marks.head(marks.height + by)
    changed.write_parquet(directory / beside["file"])
    beside["rows"] = changed.height
    (directory / MANIFEST).write_text(json.dumps(manifest))
    return directory


def _counts(said: str) -> set[int]:
    """The figures a refusal names, read back out of the sentence rather than searched for in it."""
    return {int(found.replace(",", "")) for found in re.findall(r"\d[\d,]*", said)}


class TestAStatusThatIsNotTheHeightOfItsPortIsRefused:
    """The pair is joined by position and by nothing else, so a pair that disagrees is not a pair.

    Nothing about the disagreement is loud. ``pl.concat(how="horizontal")`` pads the shorter side
    with nulls rather than failing, so a status one row short hands the last row a mark with no
    level and no reason -- which draws as a flagged-looking cell nobody can act on -- and a status
    one row long adds a row of all-null data the port does not contain, which the table draws as
    data. Both are answers, both look like the run, and neither is it. So the two counts are
    compared before they are put side by side, and the refusal names the port and both numbers,
    because "a status of the wrong height" is not something a reader can go and look at.

    Both directions, because a guard written against one is half a guard: a check for a status that
    ran out would let every over-long one through.
    """

    def test_a_status_shorter_than_its_port_is_refused_rather_than_padded_to_a_levelless_mark(
        self, written: Path, tmp_path: Path
    ) -> None:
        """Eleven marks for twelve rows: the twelfth row would otherwise be marked ``null``."""
        store = PortStore(_restated(written, tmp_path / "short", "settled.rows", -1))

        # The neighbouring port is untouched and still reads, so the refusal below is about the pair
        # that disagrees and not about a copy that did not survive being made.
        assert store.window("graded.rows", limit=12).status is not None

        with pytest.raises(PreviewError) as refused:
            store.window("settled.rows", limit=12)

        said = str(refused.value)
        assert "settled.rows" in said, f"the refusal does not say which port disagrees with its marks: {said}"
        assert _counts(said) == {11, 12}, f"the refusal does not name both heights, so neither can be found: {said}"

    def test_a_status_longer_than_its_port_is_refused_rather_than_padded_into_a_row_of_nulls(
        self, written: Path, tmp_path: Path
    ) -> None:
        """Thirteen marks for twelve rows: the thirteenth would otherwise be drawn as a row."""
        store = PortStore(_restated(written, tmp_path / "long", "settled.rows", 1))

        assert store.window("graded.rows", limit=12).status is not None

        with pytest.raises(PreviewError) as refused:
            store.window("settled.rows", limit=12)

        said = str(refused.value)
        assert "settled.rows" in said, f"the refusal does not say which port disagrees with its marks: {said}"
        assert _counts(said) == {12, 13}, f"the refusal does not name both heights, so neither can be found: {said}"


class TestTheLevelIsAVocabularyAndNotAPieceOfText:
    """What the column filters are deliberately not, because three words is not a search."""

    def test_a_level_that_is_not_one_is_refused_by_name_rather_than_matched_as_text(self, store: PortStore) -> None:
        """``warn`` is the whole case. As a substring it matches ``warning``; as a level it is nothing.

        Which is the difference between a control a reader can trust and one that happens to work: a
        substring over a level also matches a client's own value somewhere else in the row, and
        ``error`` would find every row of a column holding the word. So the level is checked against
        what the library publishes, the refusal says what the levels are, and each published level
        is proved to be accepted -- a refusal that also refused ``warning`` would pass the first
        half of this on its own.

        A port with no marks is refused in the same shape and for the same reason, because it cannot
        answer a question about levels and must not appear to. Both of the answers it could give
        instead are readings of the data when the truth is about the request: every row is plausible,
        since nothing was flagged, and no rows at all is exactly what a genuinely all-clean status
        would say.
        """
        for level in STATUS_LEVELS:
            store.window("settled.rows", limit=1, level=level)

        with pytest.raises(PreviewError) as unpublished:
            store.window("settled.rows", limit=1, level="warn")

        assert "warn" in str(unpublished.value)
        for level in STATUS_LEVELS:
            assert level in str(unpublished.value), "the refusal does not say what a level may be"

        with pytest.raises(PreviewError) as refusal:
            store.window("checked.rows", limit=12, level="warning")

        assert "checked.rows" in str(refusal.value)


# ---------------------------------------------------------------------------
# Over the wire
# ---------------------------------------------------------------------------


def _get(url: str) -> tuple[int, Any]:
    """One request, as the page makes it: the status and the parsed body, refusals included."""
    try:
        with urllib.request.urlopen(url, timeout=5) as answer:
            return answer.status, json.loads(answer.read())
    except urllib.error.HTTPError as refused:
        return refused.status, json.loads(refused.read())


@pytest.fixture
def served(tmp_path: Path, trace: PipelineTrace) -> Any:
    """The run, written and served, as the page reads it -- one flow at ``/fixture/``.

    Written again here rather than served out of the shared directory, because what these cases are
    about is the whole path from a trace to a reader: a run written by one call and served by
    another is the arrangement that actually ships.
    """
    root = tmp_path / "runs"
    _write(trace, root / "fixture")
    server = serve(root, host="127.0.0.1", port=0)
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/fixture/"
    finally:
        server.shutdown()
        server.server_close()


class TestTheServiceHandsTheMarksOverWithTheRows:
    """The page reads a window over HTTP, so a mark that stops at the store is a mark nobody sees."""

    def test_the_rows_route_answers_a_later_window_with_that_windows_marks_and_takes_a_level_of_its_own(
        self, served: str
    ) -> None:
        """One payload carries the rows and the marks, aligned, and the level is not a column filter.

        The last part is the one that cannot be seen from either side alone. ``f.level=warning``
        already means something -- narrow the column named ``level`` -- and a port carrying such a
        column is an ordinary thing for a client's data to have. Spelling the level filter that way
        would make the control work on most ports, do something entirely different on one, and be
        impossible to tell apart. So it travels as its own parameter, and the ``f.`` channel
        continues to answer about columns: asked for a column this port does not carry, it refuses.
        """
        status, window = _get(served + "api/rows?port=settled.rows&offset=8&limit=4")
        assert status == 200
        assert len(window["status"]) == len(window["rows"]) == 4
        held = _held("amount")
        ids = [row[window["columns"].index("record_id")] for row in window["rows"]]
        assert ids == ["9", "10", "11", "12"]
        for record, mark in zip(ids, window["status"], strict=True):
            if mark["level"] == "ok":
                continue
            assert mark["reason"][0]["column"] == "amount"
            assert _quoted(mark["reason"][0]["detail"]) == held[record]

        status, clean = _get(served + "api/rows?port=checked.rows&offset=0&limit=4")
        assert status == 200
        assert clean["status"] is None, "a port nothing was said about reports marks over the wire"

        status, narrowed = _get(served + "api/rows?port=settled.rows&offset=0&limit=12&level=warning")
        assert status == 200
        assert narrowed["matched"] == 6
        assert {mark["level"] for mark in narrowed["status"]} == {"warning"}

        status, refused = _get(served + f"api/rows?port=settled.rows&{FILTER_PREFIX}level=warning")
        assert status == 400, "'level' as a column filter was answered as though it were the level itself"
        assert "level" in refused["error"]


# ---------------------------------------------------------------------------
# The table draws the mark first
# ---------------------------------------------------------------------------


def _tables(markup: str) -> list[tuple[dict[str, str], str]]:
    """Every data table as (its attributes, its markup)."""
    found = []
    for tag, body in re.findall(r'<table class="data"([^>]*)>(.*?)</table>', markup, re.DOTALL):
        found.append((dict(re.findall(r'([\w-]+)="([^"]*)"', tag)), body))
    return found


def _head_cells(table: str) -> list[tuple[str, str]]:
    """The header row's cells as (attributes, contents), before the row of filters."""
    row = re.search(r"<thead><tr>(.*?)</tr>", table, re.DOTALL)
    assert row is not None, "a data table with no header row at all"
    return re.findall(r"<th\b([^>]*)>(.*?)</th>", row.group(1), re.DOTALL)


def _filter_cells(table: str) -> list[tuple[str, str]]:
    row = re.search(r'<tr class="flts">(.*?)</tr>', table, re.DOTALL)
    assert row is not None, "a data table with no row of filters"
    return re.findall(r"<th\b([^>]*)>(.*?)</th>", row.group(1), re.DOTALL)


def _table_for(panels: dict[str, str], port: str) -> str:
    """The table drawing one named port, wherever in the panels it was rendered."""
    for panel in panels.values():
        for tag, body in _tables(panel):
            if tag.get("data-port") == port:
                return body
    raise AssertionError(f"no table anywhere in the panels draws {port}")


class TestTheStatusColumnComesFirstAndOnlyWhereThereIsOne:
    """What the reader asked for, in their words: make this the first column and let me filter it."""

    def test_a_marked_ports_table_puts_the_status_column_before_the_data_and_marks_it_as_not_data(
        self, panels: dict[str, str], trace: PipelineTrace
    ) -> None:
        """First, and legible as the step talking rather than as a column the flow produced.

        A column headed ``level`` among a client's own columns reads as data: it would be taken for
        something the step computed, searched with the box under it, and carried into whatever the
        reader copies out. So it is first -- the reader's own request, and where an eye picks up a
        row -- and it is set apart in the markup rather than by its name: it wears the class the
        stylesheet and the script both key on, and it carries no dtype, which every real column's
        header does.

        The data columns are checked against the frame the run actually produced, in order, so this
        cannot pass by the status column having replaced one of them.
        """
        table = _table_for(panels, "settled.rows")
        cells = _head_cells(table)
        columns = trace.frame(SETTLED).columns

        assert len(cells) == len(columns) + 1, "the status column did not arrive beside the data's own columns"
        first, rest = cells[0], cells[1:]
        assert MARK_CLASS in first[0], f"the first header cell is {first!r}, which is one of the data's columns"
        assert "<em>" not in first[1], "the status column states a dtype, which is what a data column's header does"
        assert [re.sub(r"<em>.*</em>", "", body) for _, body in rest] == columns, (
            "the data's own columns did not come through unchanged behind the status column"
        )
        for attrs, _ in rest:
            assert MARK_CLASS not in attrs, "a data column wears the status column's own class"

    def test_only_the_ports_that_were_marked_draw_a_status_column_at_all(
        self, panels: dict[str, str], trace: PipelineTrace
    ) -> None:
        """Most ports have no marks, and a tick beside every row of every one of them means nothing.

        Pinned as an equality over every table a reader can be shown, rather than as a check on one
        clean port: the tables carrying a status column are exactly the tables drawing a port this
        run recorded marks for. That covers the two ways of getting it wrong at once -- a column on
        a port that reported nothing, and a missing column on one that did -- and it covers the
        input side, since a marked port drawn as the input of the next step is the same port and the
        same marks.

        The slices a reader picks an entity down to carry no port name and no marks with them, which
        is why they are on the far side of this equality: a slice is built in the page out of rows
        the service does not serve, so there is nothing to line marks up against.
        """
        marked = {f"{ref.step}.{ref.port}" for ref in trace.status}
        drawing = set()
        serving = set()
        for panel in panels.values():
            for tag, body in _tables(panel):
                port = tag.get("data-port")
                if port:
                    serving.add(port)
                if any(MARK_CLASS in attrs for attrs, _ in _head_cells(body)):
                    drawing.add(port)

        assert marked == {"settled.rows", "graded.rows"}, "the fixture stopped degrading anything"
        assert serving >= marked, "the marked ports are not drawn at all, so this would pass by vacancy"
        assert drawing == marked, (
            f"tables drawing a status column: {sorted(str(p) for p in drawing)}; ports with marks: {sorted(marked)}"
        )

    def test_the_search_boxes_still_name_the_cell_they_narrow_once_a_status_column_is_drawn(
        self, panels: dict[str, str]
    ) -> None:
        """The off-by-one this column would otherwise put into every filter on every marked port.

        A box says which cell of a row it narrows, and the script reads that number straight off it
        to reach ``tr.cells[c]``. Put a status cell at the front of every row and leave the numbers
        alone, and every box on a marked table narrows the column to its left: the reader types a
        party name into the party box and watches rows vanish for no reason they can see.

        Pinned as a relation between the boxes and the headers rather than as a literal offset, so
        it holds on the tables with a status column and on the tables without one, and neither can
        be satisfied by a rule that only fits the other.
        """
        every = [body for panel in panels.values() for _, body in _tables(panel)]
        marked = [body for body in every if any(MARK_CLASS in attrs for attrs, _ in _head_cells(body))]

        assert marked, "no table draws a status column, so this case is about nothing"
        for body in every:
            headers = len(_head_cells(body))
            boxes = re.findall(r'<input class="flt" data-c="(\d+)" data-col="([^"]+)"', body)
            named = [body_ for _, body_ in _head_cells(body)]
            for at, column in boxes:
                assert re.sub(r"<em>.*</em>", "", named[int(at)]) == column, (
                    f"the box for {column!r} says it narrows cell {at}, which is the cell holding "
                    f"{named[int(at)]!r}; the filter would narrow the wrong column"
                )
            assert len(boxes) == headers - (1 if body in marked else 0)

    def test_the_level_control_offers_the_published_levels_and_nothing_else(self, panels: dict[str, str]) -> None:
        """A choice of three, taken from the library, rather than a box to type a word into.

        A text box under the status column would be a substring search -- ``warn`` narrowing to
        warnings, ``o`` narrowing to everything -- and would pass the reader's word to a service
        that has to refuse it. A choice offers only what the service answers to, so the control
        cannot ask a question that has no answer, and it also cannot drift: the options are compared
        against ``STATUS_LEVELS`` itself, so a fourth level added to the library with no option here
        is a failure rather than a level nobody can filter to.

        The empty option is what "all rows" is spelled as, and it has to be first: a control that
        opens on one of the three would narrow the table before the reader touched it.
        """
        table = _table_for(panels, "settled.rows")
        control = _filter_cells(table)[0]

        assert MARK_CLASS in control[0], "the first cell of the filter row is not the status column's"
        offered = re.findall(r'<option value="([^"]*)"', control[1])
        assert offered == ["", *STATUS_LEVELS], f"the level control offers {offered}"
        assert "<input" not in control[1], "the status column offers a box to type into, which is a substring search"

        clean = _table_for(panels, "checked.rows")
        assert MARK_CLASS not in clean, "a port with nothing to narrow still offers the control"


# ---------------------------------------------------------------------------
# The mark reaches the reader
# ---------------------------------------------------------------------------


def _script(page: str) -> str:
    script = page[page.rindex("<script>") + len("<script>") : page.rindex("</script>")]
    script = re.sub(r"/\*.*?\*/", "", script, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", script)


def _block(text: str, opener: str) -> str:
    """The braced block ``opener`` begins, balanced rather than guessed at."""
    start = text.index(opener)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def _function(code: str, name: str) -> str:
    found = re.search(r"(?:async\s+)?function\s+" + re.escape(name) + r"\s*\(", code)
    assert found is not None, f"the script declares no function {name}()"
    return _block(code[found.start() :], "{")


def _window_fn(page: str) -> tuple[str, str]:
    """The code that fetches a window of rows, and that code plus the bodies of what it calls.

    Found by what it does -- it fetches, and it is about an offset rather than about a panel --
    rather than by a name, so every assertion below is about the page's behaviour and not about the
    author's choice of identifier. Whether the mark is drawn inline or in a helper is the author's
    business, which is why the callees come with it.
    """
    code = _script(page)
    for name in sorted(set(re.findall(r"(?:async\s+)?function\s+(\w+)\s*\(", code))):
        body = _function(code, name)
        if "fetch(" in body and "offset" in body:
            reached = body
            for callee in sorted(set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", body))):
                if re.search(r"(?:async\s+)?function\s+" + re.escape(callee) + r"\s*\(", code):
                    reached += _function(code, callee)
            return body, reached
    raise AssertionError("nothing in the script fetches a window of rows, so an open port never fills")


def _request_keys(region: str) -> set[str]:
    """Every parameter name the window request writes down as a literal.

    The per-column filters are built by concatenating a prefix onto a column, so they are not
    literals and do not appear here -- which is the point: what is left is the fixed vocabulary of
    the request, and a level that arrived among the column filters would be indistinguishable from
    a column called ``level``.
    """
    keys: set[str] = set()
    for hit in re.finditer(r"new\s+URLSearchParams\s*\(\s*\{([^}]*)\}", region):
        for part in hit.group(1).split(","):
            name = part.split(":")[0].strip().strip("'\"")
            if name:
                keys.add(name)
    for hit in re.finditer(r"\.(?:set|append)\(\s*(['\"])([^'\"]*)\1", region):
        keys.add(hit.group(2))
    return keys


class TestTheMarkReachesTheReader:
    """A glyph nobody can read out, on a row nobody can tie back to a value, is the count again."""

    def test_the_window_request_asks_for_a_level_by_the_name_the_service_answers_to(
        self, page: str, served: str
    ) -> None:
        """The two halves are written in different files, and only a request proves they agree.

        A renamed parameter leaves both sides looking correct: the control changes, the service
        answers about the whole port, and the reader sees a filter that appears to do nothing at
        all. So the name is read out of the script and put to the real service, and what is asserted
        is that asking with *that* name narrows to the flagged rows.

        The region is proved to be the right one before anything is read out of it -- it is the only
        place in the script that asks for a window -- because a region chosen wrongly makes every
        assertion after it a statement about some other code.
        """
        _, reached = _window_fn(page)
        code = _script(page)
        assert "api/rows" in reached, "the region examined here does not ask for a window of rows"
        assert code.count("api/rows") == 1, "a window is asked for in more than one place; this region is not the one"
        assert re.search(r"[.\[]" + MARK_CLASS + r"\b|['\"]\." + MARK_CLASS + r"\b|\bselect\b", reached), (
            "nothing in the request reaches the status column's own control, so whatever level it sends is not the "
            "reader's choice"
        )

        keys = _request_keys(reached)
        assert {"port", "offset", "limit"} <= keys, f"the window request no longer asks for a window: {sorted(keys)}"
        named = {key for key in keys - {"port", "offset", "limit"} if not key.startswith(FILTER_PREFIX)}
        assert len(named) == 1, f"the window request writes down {sorted(named)} beyond the window itself"
        asked = named.pop()
        assert FILTER_PREFIX + asked not in keys, (
            f"{asked!r} is also sent as a column filter, which is another question"
        )

        status, narrowed = _get(served + f"api/rows?port=settled.rows&limit=12&{asked}=warning")
        assert status == 200, f"the page asks for {asked!r} and the service refuses it: {narrowed}"
        assert narrowed["matched"] == 6, f"asking for {asked}=warning did not narrow to the flagged rows"

    def test_a_rows_mark_is_drawn_before_its_data_and_read_from_that_rows_own_position(self, page: str) -> None:
        """First in the row, and taken from the same position in the window the row came from.

        Both halves are the same fact seen from either end. The mark goes in before the cells, or
        the reader's first column is the data's first column and the status is somewhere in the
        middle of the table. And the mark a row is given is the one at that row's own index in the
        window's marks, because the window hands back two sequences that line up and nothing else
        joins them: any other pairing -- a running counter over a body that was not cleared, a
        lookup by a value -- draws a warning against a row that never had one.
        """
        _, reached = _window_fn(page)

        rows = re.search(r"\b(\w+)\.rows\.map\(\s*\(?([\w\s,]*?)\)?\s*=>", reached)
        assert rows is not None, "nothing maps the window's rows into rows of the table"
        params = [part.strip() for part in rows.group(2).split(",") if part.strip()]
        assert len(params) == 2, (
            f"the rows are mapped as {params}, with no index: there is then nothing to look a row's own mark up by"
        )

        subscripts = re.findall(r"\.status\s*\[\s*(\w+)\s*\]", reached)
        assert subscripts, "nothing reads a mark out of the window's marks, so no row is given one"
        assert set(subscripts) == {params[1]}, (
            f"a mark is read at {sorted(set(subscripts))} while the rows are indexed by {params[1]!r}; the two "
            "sequences line up by position and by nothing else"
        )

        assembled = _block(reached[rows.end() :], "{")
        mark = re.search(r"\.status\b", assembled)
        cells = re.search(r"\.map\(\s*\w+\s*\)|\brow\.map\(", assembled)
        assert mark is not None and cells is not None, "the row is not assembled out of a mark and its cells"
        assert mark.start() < cells.start(), (
            "the mark is put into the row after its data, so the reader's first column is the data's first column"
        )

    def test_every_level_has_its_own_drawing_and_the_sheet_tells_the_three_apart(self, page: str) -> None:
        """Three levels, three drawings, and none of them only a character.

        A level the script has no drawing for renders as neither a tick nor a warning nor a cross,
        so the row shows nothing at all and reads as fine -- which is why the drawings are compared
        against the levels the library publishes rather than against a list written down here: a
        fourth level added to ``STATUS_LEVELS`` with nothing drawn for it fails this case instead of
        going out blank.

        The stylesheet is asked the same question and has to answer differently for each, because
        three cells that differ only by a glyph are three cells that look the same at a glance and
        sound the same to a screen reader. The rules are looked for among those that name the status
        column, since the canvas already styles a warning step and an error step and would otherwise
        answer this on the script's behalf.
        """
        _, reached = _window_fn(page)

        drawings = [
            hit
            for hit in re.findall(r"\{([^{}]*)\}", reached)
            if {key.strip().strip("'\"") for key in re.findall(r"([\w'\"]+)\s*:", hit)} == set(STATUS_LEVELS)
        ]
        assert drawings, (
            f"nothing in the script draws one thing per level; the levels are {list(STATUS_LEVELS)}, and a level "
            "with no drawing renders as an empty cell, which reads as a row that is fine"
        )

        sheet = page[page.index("<style>") + len("<style>") : page.index("</style>")]
        sheet = re.sub(r"/\*.*?\*/", "", sheet, flags=re.DOTALL)
        said: dict[str, set[str]] = {level: set() for level in STATUS_LEVELS}
        for selector, declarations in re.findall(r"([^{}]+)\{([^{}]*)\}", sheet):
            if MARK_CLASS not in selector and "level" not in selector:
                continue
            for level in STATUS_LEVELS:
                if re.search(r"(?<![\w-])" + level + r"(?![\w-])", selector):
                    said[level].add(declarations.strip())

        missing = sorted(level for level, rules in said.items() if not rules)
        assert not missing, f"the sheet says nothing about the status column for {missing}"
        assert len({frozenset(rules) for rules in said.values()}) == len(STATUS_LEVELS), (
            f"the sheet draws two levels identically: {said}"
        )

    def test_the_reason_reaches_the_reader_as_the_column_it_names_and_what_that_cell_held(self, page: str) -> None:
        """The sentence the reader came for, without leaving the panel they are looking at.

        A mark that says only "something happened here" sends a reader back to comparing two tables
        by eye, which is the work this whole channel exists to remove -- and a mark that says it
        only in colour says nothing to anyone using a screen reader. So the cell carries the level
        as text the browser can announce, and the reason as the column that degraded and what the
        cell held: ``held 'Canadian Business Growth Fund', which did not read as Float64``.

        Both halves of the reason are required, because either alone is the old warning again. The
        column without the detail is a count that names a column; the detail without the column is a
        sentence a reader cannot act on when a step read four columns.
        """
        _, reached = _window_fn(page)

        spoken = re.findall(
            r"\.(?:title|ariaLabel)\s*=([^;]*)|setAttribute\(\s*['\"](?:title|aria-label)['\"],([^;]*)", reached
        )
        text = " ".join(part for hit in spoken for part in hit)
        assert text.strip(), (
            "the mark carries no text a browser can announce, so the three levels differ by a character alone and a "
            "reader who cannot see it is told nothing"
        )
        assert re.search(r"\.level\b", text), "the announced text does not name the level"
        assert re.search(r"\bcolumn\b", text), "the announced text does not name the column that degraded"
        assert re.search(r"\bdetail\b", text), "the announced text does not say what the cell held"

    def test_the_character_drawn_for_a_flagged_row_is_not_the_one_drawn_for_a_clean_one(
        self, page: str, store: PortStore
    ) -> None:
        """Which glyph goes with which level, rather than that three glyphs exist.

        The case beside this one checks that every published level has a drawing and that the sheet
        tells the three apart; neither ever reads the drawings themselves. So a pairing with the
        tick and the cross the wrong way round passes both -- and draws every flagged row as clean
        and every clean row as failed, in the first column the reader looks at, which is worse than
        drawing nothing: a reader who trusts the column reads the twelve rows exactly backwards.

        Pinned as a property of the characters rather than as the author's literals, so a different
        tick (``\u2714``) or a different cross is free to be chosen and a swap is not: the mark for
        ``ok`` has to read as a tick, and no level that is not ``ok`` may draw one. Then the pairing
        is walked back through a real window of a real run -- the rows whose amount did not read as
        a figure against the rows whose did -- because the pairing is only ever reached by way of a
        level a step actually wrote down.
        """
        drawings = _drawings(page)

        assert len(set(drawings.values())) == len(STATUS_LEVELS), (
            f"two levels draw the same character, so the column cannot tell them apart: {drawings}"
        )
        assert all(mark.strip() for mark in drawings.values()), f"a level draws nothing at all: {drawings}"

        ticks = {level for level, mark in drawings.items() if "CHECK" in unicodedata.name(mark)}
        assert ticks == {"ok"}, (
            f"a tick is what a reader reads as 'this row is fine', and {sorted(ticks)} draw one: {drawings}"
        )

        window = store.window("settled.rows", limit=12)
        held = _held("amount")
        drawn = {True: set(), False: set()}
        for row, mark in zip(window.rows, window.status, strict=True):
            record = _column_of(row, window, "record_id")
            drawn[_is_a_number(held[record])].add(drawings[mark["level"]])

        assert drawn[True] == {drawings["ok"]}, "a row whose amount read as a figure is drawn as something else"
        assert drawn[False], "this fixture flagged nothing, so it cannot tell a flagged row from a clean one"
        assert not (drawn[True] & drawn[False]), (
            f"a row the step emptied is drawn with the same character as a row it left alone: {drawn}"
        )


def _drawings(page: str) -> dict[str, str]:
    """The character the window's script draws for each published level, read out of the pairing.

    Found by its keys, the way the case above it finds the block at all, and then read for its
    values -- which is the whole difference between the two: a block whose keys are the levels says
    only that somebody thought about all three.
    """
    _, reached = _window_fn(page)
    found = [
        dict(re.findall(r"['\"]?(\w+)['\"]?\s*:\s*['\"](.*?)['\"]", hit))
        for hit in re.findall(r"\{([^{}]*)\}", reached)
        if {key.strip().strip("'\"") for key in re.findall(r"([\w'\"]+)\s*:", hit)} == set(STATUS_LEVELS)
    ]
    assert len(found) == 1, f"the script draws one thing per level in {len(found)} places, so which one draws a row?"
    pairs = {level: _character(mark) for level, mark in found[0].items()}
    assert set(pairs) == set(STATUS_LEVELS), f"the drawings do not pair with the published levels: {pairs}"
    return pairs


def _character(mark: str) -> str:
    """One drawn mark, with a JavaScript escape resolved to the character a browser would show."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda found: chr(int(found.group(1), 16)), mark)


def _is_a_number(text: str) -> bool:
    """Whether a source value would read as a figure, which is what decides its level."""
    try:
        float(text)
    except ValueError:
        return False
    return True
