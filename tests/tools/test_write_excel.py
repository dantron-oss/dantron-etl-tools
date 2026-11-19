"""Tests for the WriteExcel tool: many frames, one workbook, one save, and nothing else touched.

WriteExcel takes one input port per tab its configuration names, writes each frame to its tab of
one XLSX workbook, and produces nothing. It is the catalog's first tool of either kind: the first
that works its input ports out from its own configuration, and the first that is terminal.

Eleven tabs is one step reading eleven named frames. The alternative -- eleven chained steps, each
reopening the file -- is the thing this tool exists to stop, and it is not merely slower. A tool
that reopens a workbook per tab has to decide, eleven times, what to do about the tabs already
there, and a template full of pivots and formats gets a chance to be rebuilt on each of those
passes.

The rulings pinned here, each argued rather than assumed:

**The ports come from ``sheets``.** Two steps of this tool with different ``sheets`` read different
ports. ``input_ports`` is empty, and has to be: a tool that declares static ports and derives them
as well answers the question two ways, and the library refuses it. ``sheets`` is required and
carries no default, because derivation reads the configuration as an author wrote it, before any
blank is filled in -- a port list behind a default would be invisible to the loader.

**It is terminal.** A workbook writer has nothing honest to hand on: handing back some or all of
eleven inputs would be a fan-out this tool invented, and handing back a receipt would be shape
without information. So it produces no ports at all.

**Writing into a template leaves every other tab alone.** This is the point of the tool. A
delivered pack is a curated workbook -- pivots, charts, formats, notes somebody wrote -- with a few
data tabs the flow refreshes. The tool loads that workbook, replaces the tabs it was told about,
and saves the whole thing. A tab nobody named comes back with its values, its formulas, its
formatting and its column widths, and the workbook keeps its workbook-level parts. That last one is
the strongest property these tests can pin without building a pivot table by hand: a defined name
lives in the workbook part rather than in any sheet, so a writer that rebuilt the file from scratch
loses it while every sheet still looks right.

**The template is copied to ``path``, never written in place.** A template overwritten by its own
first run is gone, and the next run has nothing to start from. So the template is read and never
written, and ``path`` equal to ``template`` is refused outright rather than left to be discovered
once.

**``overwrite_sheets`` governs the tab, not the file.** A tab already in the workbook is refused until an
author says in writing that overwriting it is intended, and a tab nobody named is never at risk
either way.

**A port with no rows writes its tab with the headers and nothing under them.** A missing tab reads
as a broken report. An empty tab reads as "nothing this month".

**What a cell cannot hold is refused.** A whole number past 2**53 - 1, a Decimal, a Time, a
Duration, a list, a struct, a frame taller than a sheet. The numbers are the ones in
``_spreadsheet.py``, imported below rather than written down twice.

Vocabulary is deliberately generic -- key, amount, code -- because the library carries no caller's
nouns.
"""

import datetime as dt
import os
import posixpath
import re
import zipfile
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import openpyxl
import polars as pl
import pytest
from openpyxl.styles import Font
from openpyxl.workbook.defined_name import DefinedName

from dantron_etl.column_types import DECLARED_TYPES
from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    DataError,
    Ports,
    check_tool,
    expression_fields,
    input_ports_for,
    ports_from_config,
    run_tool,
)
from dantron_etl.tools._spreadsheet import MAX_DATA_ROWS, MAX_EXACT_INTEGER
from dantron_etl.tools.write_excel import WriteExcel

# ---------------------------------------------------------------------------
# Test data and helpers
# ---------------------------------------------------------------------------


def _rows(marker: str) -> pl.DataFrame:
    """Three rows carrying ``marker``, so a test can tell which frame landed on which tab."""
    return pl.DataFrame(
        {
            "key": [f"{marker}1", f"{marker}2", f"{marker}3"],
            "amount": [10.25, -5.50, 1234.56],
            "code": [marker, marker, marker],
        }
    )


def _write(frames: Mapping[str, pl.DataFrame], config: Mapping[str, Any]) -> None:
    """Put ``frames`` through the boundary, and check the tool really is terminal.

    The assertion belongs in the helper rather than in one test, because every call here would
    otherwise quietly accept a tool that grew an output port later.
    """
    result = run_tool(WriteExcel(), Ports(dict(frames)), dict(config))
    assert result.names == (), "a terminal sink produces no ports at all"


def _read_back(path: Path, sheet: str) -> pl.DataFrame:
    """One tab as the spreadsheet library itself sees it, not as this library's reader sees it.

    Deliberately the raw reader rather than the paired ReadExcel tool: what is under test is the
    artefact left on disk, and reading it back through the other tool would make every assertion
    below depend on that tool's own rulings as well.
    """
    return pl.read_excel(path, sheet_name=sheet, drop_empty_cols=False, raise_if_empty=False)


def _tab_names(path: Path) -> list[str]:
    """The tabs of the workbook, in the order a person opening it sees them."""
    book = openpyxl.load_workbook(path)
    try:
        return list(book.sheetnames)
    finally:
        book.close()


def _template(path: Path, *, tabs: tuple[str, ...] = ("Notes", "Summary")) -> Path:
    """A curated workbook of the kind a delivered pack starts life as.

    The ``Notes`` tab is the one the flow never names, and it carries one of everything a writer
    that rebuilt the file from scratch would lose: a value, a formula, a bold font, a column width.
    The defined name is the strongest of the five, because it lives in the workbook part rather than
    in any sheet -- a rebuild that copied every tab across faithfully would still drop it.
    """
    book = openpyxl.Workbook()
    book.remove(book.active)
    for tab in tabs:
        sheet = book.create_sheet(tab)
        sheet["A1"] = f"carried over: {tab}"
        sheet["A1"].font = Font(bold=True)
        sheet["B2"] = 42
        sheet["C3"] = "=SUM(B2:B2)"
        sheet.column_dimensions["A"].width = 30
    book.defined_names["carried_total"] = DefinedName("carried_total", attr_text=f"'{tabs[0]}'!$B$2")
    book.save(path)
    book.close()
    return path


def _notes_survived(path: Path, tab: str = "Notes") -> None:
    """Everything the untouched tab held, still held, plus the workbook part nothing named.

    Five separate properties, deliberately asserted together: they are one claim -- the workbook was
    opened and saved, not rebuilt -- and splitting them would turn one failure into five.
    """
    book = openpyxl.load_workbook(path)
    try:
        assert tab in book.sheetnames, "a tab the flow never named is gone from the pack"
        sheet = book[tab]
        assert sheet["A1"].value == f"carried over: {tab}", "the untouched tab lost its values"
        assert sheet["B2"].value == 42
        assert sheet["C3"].value == "=SUM(B2:B2)", "the untouched tab's formula was flattened or dropped"
        assert sheet["A1"].font.bold is True, "the untouched tab lost its formatting"
        assert sheet.column_dimensions["A"].width == 30, "the untouched tab lost its column widths"
        assert "carried_total" in book.defined_names, (
            "the workbook's own parts were rebuilt rather than carried over; a pivot table would have gone the same way"
        )
    finally:
        book.close()


PIVOT_TEMPLATE = Path(__file__).resolve().parents[1] / "fixtures" / "workbook_with_pivot.xlsx"
"""A small workbook with one real pivot table, built by ``tests/fixtures/make_workbook_with_pivot.py``.

Two tabs, four data rows, one pivot and one pivot cache. It is here because a pivot table is the
thing this tool's whole design protects, and until now no test opened one. A defined name is not a
substitute: it lives in ``xl/workbook.xml``, which round-trips faithfully, so it says nothing at all
about the pivot parts beside it.
"""

_SPREADSHEET_XML = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

_PIVOT_PART_KINDS = ("pivotCacheDefinition", "pivotCacheRecords", "pivotTable")


def _xml_parts(path: Path) -> list[str]:
    """The XML parts inside the file, leaving out the relationship files."""
    with zipfile.ZipFile(path) as archive:
        return [name for name in archive.namelist() if name.endswith(".xml") and "_rels" not in name]


def _pivot_parts(path: Path) -> dict[str, int]:
    """How many parts of each pivot kind the file holds.

    Counted, not named. A writer is entitled to renumber a part -- pivotCacheDefinition1 becoming
    pivotCacheDefinition2 -- and has lost nothing by doing it. A test matching names would call that
    a failure, so this one counts instead.
    """
    names = _xml_parts(path)
    return {kind: sum(1 for name in names if kind in name) for kind in _PIVOT_PART_KINDS}


def _pivot_content_types(path: Path) -> int:
    """How many pivot content types the manifest declares.

    A part sitting in the zip but undeclared here is a part nothing will open. The file is repaired
    on the way in, and the pivot is gone.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = archive.read("[Content_Types].xml").decode("utf-8")
    return len(re.findall(r'ContentType="[^"]*pivot[^"]*"', manifest))


def _cache_definitions(path: Path) -> list[dict[str, Any]]:
    """One entry per pivot cache: what it claims to hold, and how much of it is really there.

    ``record_count`` is the claim. ``shared_item_values`` counts the cached values under every
    ``sharedItems`` block. Those values are what fills a pivot's filter dropdown before anybody
    refreshes it, so a cache that keeps the claim and drops the values delivers empty filters.

    ``extensions`` counts the ``ext`` blocks. That is where a spreadsheet program keeps the parts of
    a pivot that postdate the original file format, a slicer link among them.
    """
    definitions: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(n for n in _xml_parts(path) if "pivotCacheDefinition" in n):
            raw = archive.read(name)
            root = ElementTree.fromstring(raw)
            blocks = list(root.iter(f"{_SPREADSHEET_XML}sharedItems"))
            definitions.append(
                {
                    "record_count": root.get("recordCount"),
                    "shared_item_blocks": len(blocks),
                    "shared_item_values": sum(len(block) for block in blocks),
                    "extensions": len(list(root.iter(f"{_SPREADSHEET_XML}ext"))),
                    "bytes": len(raw),
                }
            )
    return definitions


def _dangling_relationships(path: Path) -> list[tuple[str, str]]:
    """Every relationship pointing at a part that is not in the file.

    One of these is what makes a spreadsheet program offer to repair a workbook, and a repaired
    workbook is one with the pivot taken out.
    """
    dangling: list[tuple[str, str]] = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        for rels in sorted(name for name in names if name.endswith(".rels")):
            base = posixpath.dirname(posixpath.dirname(rels))
            for relationship in ElementTree.fromstring(archive.read(rels)):
                if relationship.get("TargetMode") == "External":
                    continue
                target = relationship.get("Target", "")
                resolved = target[1:] if target.startswith("/") else posixpath.normpath(posixpath.join(base, target))
                if resolved not in names:
                    dangling.append((rels, target))
    return dangling


# ---------------------------------------------------------------------------
# Several frames become several tabs
# ---------------------------------------------------------------------------


class TestSeveralFramesBecomeSeveralTabs:
    """One step, one workbook, one tab per named port -- and the values survive the trip."""

    def test_every_named_port_lands_on_its_own_tab_in_the_order_the_sheets_name_them(
        self,
        tmp_path: Path,
    ) -> None:
        """Three frames in, three tabs out, each holding its own rows and nobody else's.

        Tab order is pinned on purpose. A pack is read by a person, top to bottom, and the order of
        its tabs is the order the author wrote them in -- not whatever order a dictionary happened
        to iterate in. A writer that got the rows right and the order wrong is caught here and
        nowhere else.
        """
        target = tmp_path / "pack.xlsx"
        # Named in reverse alphabetical order on purpose. Written as first, second, third, this test
        # passed whether the tabs were paired with the ports as authored or with the ports sorted,
        # because those two happen to be the same list.
        frames = {"zulu": _rows("a"), "yankee": _rows("b"), "xray": _rows("c")}

        _write(
            frames,
            {
                "path": str(target),
                "sheets": [
                    {"port": "zulu", "sheet": "Figures"},
                    {"port": "yankee", "sheet": "Detail"},
                    {"port": "xray", "sheet": "Notes"},
                ],
            },
        )

        assert _tab_names(target) == ["Figures", "Detail", "Notes"]
        for tab, marker in (("Figures", "a"), ("Detail", "b"), ("Notes", "c")):
            back = _read_back(target, tab)
            assert back.columns == ["key", "amount", "code"], f"{tab} lost its header"
            assert back["code"].to_list() == [marker] * 3, f"{tab} holds another port's rows"
            assert back["key"].to_list() == [f"{marker}1", f"{marker}2", f"{marker}3"]

    @pytest.mark.parametrize(
        ("column", "expected_values"),
        [
            pytest.param(
                pl.Series("amount", [10.25, -5.5, 1234.56]),
                [10.25, -5.5, 1234.56],
                id="a_figure_to_two_decimals_is_exact_both_ways",
            ),
            pytest.param(pl.Series("count", [1, 0, -7]), [1, 0, -7], id="a_whole_number_stays_whole"),
            pytest.param(pl.Series("code", ["A", "B", "A"]), ["A", "B", "A"], id="text_stays_text"),
            pytest.param(
                pl.Series("settled", [True, False, True]),
                [True, False, True],
                id="a_truth_value_stays_a_truth_value",
            ),
            pytest.param(
                pl.Series("on", [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 12, 1)]),
                [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 12, 1)],
                id="a_date_comes_back_a_date_and_not_a_serial_number",
            ),
            pytest.param(
                pl.Series("amount", [10.25, None, 1234.56]),
                [10.25, None, 1234.56],
                id="a_missing_figure_comes_back_missing",
            ),
            pytest.param(
                pl.Series("note", ["seen", "", None]),
                ["seen", None, None],
                id="an_empty_string_and_a_missing_value_are_the_same_blank_cell",
            ),
        ],
    )
    def test_a_value_written_to_a_tab_comes_back_as_the_kind_of_thing_it_was(
        self,
        tmp_path: Path,
        column: pl.Series,
        expected_values: list[Any],
    ) -> None:
        """The whole point of a typed destination, and it has to hold on every tab, not the first.

        The date case is the one that catches a real defect. A spreadsheet stores a date as a count
        of days, and the only thing telling a cell it holds a date rather than the number 46053 is
        the number format written beside it. A writer that applies the format while creating the
        first tab and forgets it on the rest produces a pack where one tab's dates compare and the
        others silently compare serial numbers, so the column is written to two tabs here.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1", "r2", "r3"]}).with_columns(column)

        _write(
            {"one": frame, "two": frame},
            {
                "path": str(target),
                "sheets": [{"port": "one", "sheet": "First"}, {"port": "two", "sheet": "Second"}],
            },
        )

        for tab in ("First", "Second"):
            back = _read_back(target, tab)
            assert back[column.name].to_list() == expected_values, f"{tab} changed the values"
            assert back.schema[column.name] == pl.Series(expected_values).dtype, (
                f"{tab} changed what kind of thing the column holds"
            )

    def test_a_port_with_no_rows_writes_its_tab_with_the_headers_and_nothing_under_them(
        self,
        tmp_path: Path,
    ) -> None:
        """An empty tab, never an absent one -- and the tabs beside it are unaffected.

        Three things could happen to an empty frame and only one of them is defensible. Leaving the
        tab out delivers a pack whose contents page does not match its tabs, and a reader cannot
        tell a month with nothing in it from a run that half failed. Refusing stops a flow that is
        otherwise correct, and AssertRows already exists for a flow that wants emptiness to stop the
        run, said in writing. So the tab is written with its header and no rows, which reads as
        "nothing this month" to the person who opens it.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"full": _rows("a"), "empty": _rows("b").head(0)},
            {
                "path": str(target),
                "sheets": [{"port": "full", "sheet": "Figures"}, {"port": "empty", "sheet": "Adjustments"}],
            },
        )

        assert _tab_names(target) == ["Figures", "Adjustments"], "an empty port still gets its tab"
        empty = _read_back(target, "Adjustments")
        assert empty.columns == ["key", "amount", "code"], "the header is what says which month was empty"
        assert empty.height == 0
        assert _read_back(target, "Figures").height == 3, "the empty port disturbed the tab beside it"

    @pytest.mark.parametrize(
        ("column", "expected_format"),
        [
            pytest.param(
                pl.Series("on", [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 12, 1)]),
                "yyyy-mm-dd",
                id="a_date",
            ),
            pytest.param(
                pl.Series(
                    "at",
                    [
                        dt.datetime(2026, 1, 31, 13, 45, 30),
                        dt.datetime(2026, 2, 28, 0, 0, 0),
                        dt.datetime(2026, 12, 1, 23, 59, 59),
                    ],
                ),
                "yyyy-mm-dd hh:mm:ss",
                id="an_instant",
            ),
        ],
    )
    def test_the_number_format_on_a_data_cell_is_the_one_this_tool_declares(
        self,
        tmp_path: Path,
        column: pl.Series,
        expected_format: str,
    ) -> None:
        """Read the format code itself, because reading the value back cannot tell who wrote it.

        The spreadsheet library puts a date format on a cell of its own accord, the moment a date is
        put in a cell whose format is still General. So a tool that never formatted anything at all
        would still hand back dates that read as dates, and a round trip through a reader would
        report no difference. Delete this tool's formatting line and every value test here still
        passes.

        The format code is what separates the two. The instant case is where they visibly differ:
        this tool declares hours as ``hh``, and the format the library fills in on its own says
        ``h``. Asserting the declared code therefore says the tool formatted the cell, not that
        something did.

        Asked of the second tab, because the defect worth catching is a writer that formats while it
        builds the first tab and forgets on the rest. The key column is checked in the same breath:
        a tool that formatted every cell would be making a claim about the pack it is not entitled
        to make.
        """
        frame = pl.DataFrame({"key": ["r1", "r2", "r3"]}).with_columns(column)
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": frame, "two": frame},
            {
                "path": str(target),
                "sheets": [{"port": "one", "sheet": "First"}, {"port": "two", "sheet": "Second"}],
            },
        )

        book = openpyxl.load_workbook(target)
        try:
            for tab in ("First", "Second"):
                assert book[tab]["B2"].number_format == expected_format, (
                    f"{tab} carries a format this tool did not write"
                )
                assert book[tab]["A2"].number_format == "General", "a text column was given a format nobody asked for"
        finally:
            book.close()

    def test_an_instant_lands_on_every_tab_keeping_its_moment_and_losing_only_its_resolution(
        self,
        tmp_path: Path,
    ) -> None:
        """A change rather than a preservation, written down so nobody finds it by comparing schemas.

        A cell has no notion of microseconds. The moment survives to the second and the resolution
        does not, so a timestamp column arrives at the far side as milliseconds. That is a loss this
        destination imposes and the tool cannot avoid. Said here rather than discovered three steps
        later by somebody wondering why two schemas stopped matching.

        It is not in the round-trip cases above for the same reason: those derive the expected type
        from the expected values, and this is the one column whose type on the way out is not the
        type on the way in.
        """
        moments = [
            dt.datetime(2026, 1, 31, 13, 45, 30),
            dt.datetime(2026, 2, 28, 0, 0, 0),
            dt.datetime(2026, 12, 1, 23, 59, 59),
        ]
        frame = pl.DataFrame({"key": ["r1", "r2", "r3"], "at": moments})
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": frame, "two": frame},
            {
                "path": str(target),
                "sheets": [{"port": "one", "sheet": "First"}, {"port": "two", "sheet": "Second"}],
            },
        )

        for tab in ("First", "Second"):
            back = _read_back(target, tab)
            assert back["at"].to_list() == moments, f"{tab} moved the instant"
            assert back.schema["at"] == pl.Datetime("ms"), f"{tab} did not write a timestamp as a timestamp"


# ---------------------------------------------------------------------------
# The template
# ---------------------------------------------------------------------------


class TestWritingIntoATemplate:
    """The reason the tool exists: refresh a few tabs of a curated pack and touch nothing else."""

    def test_a_tab_the_flow_never_names_comes_back_exactly_as_it_was(self, tmp_path: Path) -> None:
        """The preservation claim, pinned as far as a test without a pivot table can pin it.

        A pivot table cannot reasonably be built here, so what is asserted instead is the property a
        pivot would depend on: the workbook was loaded and saved rather than rebuilt. A rebuild
        shows up in five places at once -- values, formulas, fonts, column widths and the
        workbook-level defined name -- and the last of those is the one no per-tab copy can fake.

        The two written tabs are checked in the same test on purpose. Preservation that came at the
        price of not actually writing anything would otherwise pass.
        """
        template = _template(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _rows("a"), "two": _rows("b")},
            {
                "path": str(target),
                "template": str(template),
                "sheets": [{"port": "one", "sheet": "Figures"}, {"port": "two", "sheet": "Detail"}],
            },
        )

        _notes_survived(target)
        assert _tab_names(target) == ["Notes", "Summary", "Figures", "Detail"], (
            "the template's own tabs keep their order, and new tabs are appended in the order the sheets name them"
        )
        assert _read_back(target, "Figures")["code"].to_list() == ["a", "a", "a"]
        assert _read_back(target, "Detail")["code"].to_list() == ["b", "b", "b"]

    def test_the_template_is_copied_to_the_path_and_never_written_to_itself(self, tmp_path: Path) -> None:
        """A template overwritten by its own first run is gone, and the second run has nothing.

        The decision: ``template`` is read-only, always, and ``path`` is the only file this tool
        writes. The alternative -- writing in place and letting an author point ``path`` back at the
        template -- destroys the curated workbook on the first run of a flow that was one typo away
        from being right, and no run after it can tell what was lost.
        """
        template = _template(tmp_path / "template.xlsx")
        before = template.read_bytes()
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _rows("a")},
            {
                "path": str(target),
                "template": str(template),
                "sheets": [{"port": "one", "sheet": "Figures"}],
            },
        )

        assert template.read_bytes() == before, "the template was written to; next month it is gone"
        assert "Figures" not in _tab_names(template), "the template grew a tab"
        assert "Figures" in _tab_names(target), "the pack never got the tab"

    @pytest.mark.parametrize(
        ("use_path_as_template", "template_exists", "named"),
        [
            pytest.param(True, True, "template", id="a_path_equal_to_the_template_would_destroy_it"),
            pytest.param(False, False, "template", id="a_template_that_is_not_there_is_not_a_template"),
        ],
    )
    def test_a_template_that_cannot_be_used_is_refused_before_anything_is_written(
        self,
        tmp_path: Path,
        use_path_as_template: bool,
        template_exists: bool,
        named: str,
    ) -> None:
        """Both cases are answerable from the configuration and the filesystem, so both refuse early.

        Writing into the template is refused rather than merely discouraged, because the tool cannot
        distinguish "the author meant it" from "the author copied the line above and forgot to
        change one of the two paths". The cost of being wrong differs by everything: a refusal costs
        one edit, and the other outcome costs the template.

        A missing template is refused rather than quietly treated as "no template", because a
        misspelled template path would otherwise deliver a pack containing only the data tabs, which
        opens cleanly and looks like a formatting problem rather than a wiring one.
        """
        template = _template(tmp_path / "template.xlsx") if template_exists else tmp_path / "absent.xlsx"
        target = template if use_path_as_template else tmp_path / "pack.xlsx"
        before = template.read_bytes() if template_exists else None

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _rows("a")},
                {
                    "path": str(target),
                    "template": str(template),
                    "sheets": [{"port": "one", "sheet": "Figures"}],
                },
            )

        message = str(caught.value)
        assert named in message, f"the refusal does not name what is wrong: {message}"
        assert str(template) in message, f"the refusal does not name the file: {message}"
        if before is not None:
            assert template.read_bytes() == before, "a refused write must not have touched the template"

    @pytest.mark.parametrize(
        "spelling",
        [
            pytest.param("detour", id="a_path_that_goes_up_a_directory_and_back_down_again"),
            pytest.param("relative", id="one_of_the_two_written_relative_to_where_the_run_started"),
            pytest.param("symlink", id="a_link_standing_in_for_the_template"),
        ],
    )
    def test_a_path_naming_the_template_by_any_other_spelling_is_refused_too(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        spelling: str,
    ) -> None:
        """Same file, different spelling, same refusal -- because the file is what cannot be overwritten.

        The rule is about a file on disk, not about two strings. Three of these name the template as
        surely as typing its path twice does, and each is a thing an author really writes: a path
        assembled from a directory and a name that walks back up out of a subfolder; one path written
        out in full and the other relative to where the run started; a link somebody made so two
        flows could share one template.

        How the refusal decides is not pinned here, and deliberately so. Resolving both paths and
        comparing them, or asking the filesystem whether the two name the same file, are both fine
        answers, and they differ on cases nobody has yet had to rule on. What is pinned is the
        outcome: it refuses, and the template is byte-for-byte what it was.
        """
        template = _template(tmp_path / "template.xlsx")
        before = template.read_bytes()

        if spelling == "detour":
            (tmp_path / "sub").mkdir()
            named_path = str(tmp_path / "sub" / ".." / "template.xlsx")
            named_template = str(template)
        elif spelling == "relative":
            monkeypatch.chdir(tmp_path)
            named_path = str(template)
            named_template = "template.xlsx"
        else:
            link = tmp_path / "link.xlsx"
            try:
                link.symlink_to(template)
            except OSError:  # pragma: no cover - a platform that will not make links
                pytest.skip("this platform does not allow symlinks")
            named_path = str(link)
            named_template = str(template)

        # Caught rather than asserted with pytest.raises, so the failure reported first is the
        # destructive one. A run that was allowed through has already overwritten the template by
        # the time it returns, and that is what a reader of this failure needs told.
        refusal: ConfigError | None = None
        try:
            _write(
                {"one": _rows("a")},
                {
                    "path": named_path,
                    "template": named_template,
                    "sheets": [{"port": "one", "sheet": "Figures"}],
                },
            )
        except ConfigError as error:
            refusal = error

        assert template.read_bytes() == before, (
            "the template was overwritten by a path that named it another way; next run has nothing to start from"
        )
        assert refusal is not None, "the write went ahead against the template"
        assert "template" in str(refusal), f"the refusal does not say what is wrong: {refusal}"

    def test_a_hard_link_to_the_template_is_refused_and_leaves_it_untouched(self, tmp_path: Path) -> None:
        """A fourth name for the template, and the one a resolved path cannot see.

        A hard link is two directory entries for one file. Neither entry points at the other, the
        way a symlink does. Both resolve to themselves. So a guard that resolves the two paths and
        compares them sees two different files, lets the write through, and the template is gone.

        The rule is the same as for the three spellings above: the file is what may not be
        overwritten. How the tool answers the question is not pinned here. Resolving has to stay the
        first check, because ``path`` is usually the pack about to be created and there is nothing
        on disk to ask about. What is pinned is the outcome: it refuses, and the template is
        byte-for-byte what it was.
        """
        template = _template(tmp_path / "template.xlsx")
        before = template.read_bytes()
        link = tmp_path / "pack.xlsx"
        try:
            os.link(template, link)
        except (OSError, NotImplementedError):  # pragma: no cover - a platform without hard links
            pytest.skip("this platform or filesystem does not allow hard links")
        if not template.samefile(link):  # pragma: no cover - a filesystem that copied instead
            pytest.skip("this filesystem copied the template rather than linking it")

        # Caught rather than asserted with pytest.raises, so the failure reported first is the
        # destructive one. A run that was allowed through has already overwritten the template by
        # the time it returns, and that is what a reader of this failure needs told.
        refusal: ConfigError | None = None
        try:
            _write(
                {"one": _rows("a")},
                {
                    "path": str(link),
                    "template": str(template),
                    "sheets": [{"port": "one", "sheet": "Figures"}],
                },
            )
        except ConfigError as error:
            refusal = error

        assert template.read_bytes() == before, (
            "the template was overwritten through a hard link to it; next run has nothing to start from"
        )
        assert refusal is not None, "the write went ahead against the template"
        assert "template" in str(refusal), f"the refusal does not say what is wrong: {refusal}"

    def test_a_pack_that_is_not_there_yet_is_written_from_a_template_that_is(self, tmp_path: Path) -> None:
        """The ordinary run, pinned here so a stricter guard cannot take it away.

        Almost every run looks like this. The template is on disk and the pack is not: the pack is
        the file this run creates. Asking the filesystem whether two paths are one file has no
        answer until both files are there. A guard that asked anyway, and refused or fell over when
        the pack was missing, would stop every flow this tool has.
        """
        template = _template(tmp_path / "template.xlsx")
        before = template.read_bytes()
        target = tmp_path / "pack.xlsx"
        assert not target.exists(), "the pack must not exist yet, or this test pins nothing"

        _write(
            {"one": _rows("a")},
            {
                "path": str(target),
                "template": str(template),
                "sheets": [{"port": "one", "sheet": "Figures"}],
            },
        )

        assert _read_back(target, "Figures")["code"].to_list() == ["a", "a", "a"]
        _notes_survived(target)
        assert template.read_bytes() == before, "the template was written to on an ordinary run"

    def test_a_pack_spelled_the_same_awkward_way_is_still_written_when_it_is_not_the_template(
        self,
        tmp_path: Path,
    ) -> None:
        """The other half of the rule: refusing every awkward path would be its own defect.

        A path with a detour in it is ordinary -- it falls out of joining a configured directory to a
        filename. Only the ones that land on the template are refused. This is here so the guard
        cannot be satisfied by refusing anything that looks unusual.
        """
        template = _template(tmp_path / "template.xlsx")
        (tmp_path / "sub").mkdir()

        _write(
            {"one": _rows("a")},
            {
                "path": str(tmp_path / "sub" / ".." / "pack.xlsx"),
                "template": str(template),
                "sheets": [{"port": "one", "sheet": "Figures"}],
            },
        )

        target = tmp_path / "pack.xlsx"
        assert target.exists(), "an ordinary path with a detour in it was refused"
        assert _read_back(target, "Figures")["code"].to_list() == ["a", "a", "a"]
        _notes_survived(target)


# ---------------------------------------------------------------------------
# overwrite_sheets, which is about a tab
# ---------------------------------------------------------------------------


class TestReplaceGovernsTheTabNotTheFile:
    """The core difference from the single-sheet sink, where the same word governs the whole file."""

    @pytest.mark.parametrize(
        ("overwrite", "outcome"),
        [
            pytest.param(None, "refused", id="a_tab_already_there_is_not_overwritten_by_default"),
            pytest.param(False, "refused", id="nor_when_the_author_said_not_to"),
            pytest.param(True, "replaced", id="only_when_the_author_said_so"),
        ],
    )
    def test_a_tab_already_in_the_template_is_replaced_only_when_the_author_said_so(
        self,
        tmp_path: Path,
        overwrite: bool | None,
        outcome: str,
    ) -> None:
        """The unrecoverable mistake this tool can make is at tab granularity, so the permission is too.

        ``write_excel`` guards a file, because a file is all it writes. This tool writes into a pack
        that somebody else's work is also in, so the thing worth guarding is smaller: the tab that
        already holds last quarter's hand-built summary. A replaced tab keeps its position among the
        others, because a tab that moved to the end of the pack is a tab a reader will report as
        missing.
        """
        template = _template(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"
        config: dict[str, Any] = {
            "path": str(target),
            "template": str(template),
            "sheets": [{"port": "one", "sheet": "Summary"}, {"port": "two", "sheet": "Figures"}],
        }
        if overwrite is not None:
            config["overwrite_sheets"] = overwrite

        if outcome == "refused":
            with pytest.raises(ConfigError) as caught:
                _write({"one": _rows("a"), "two": _rows("b")}, config)

            message = str(caught.value)
            assert "Summary" in message, f"the refusal does not name the tab it declined to destroy: {message}"
            assert "overwrite_sheets" in message, f"the refusal does not name the keystroke that authorises it: {message}"
            assert not target.exists(), "a refused write left a pack behind"
            return

        _write({"one": _rows("a"), "two": _rows("b")}, config)

        assert _tab_names(target) == ["Notes", "Summary", "Figures"], "the replaced tab moved"
        assert _read_back(target, "Summary")["code"].to_list() == ["a", "a", "a"], "the tab was not replaced"
        _notes_survived(target)

    def test_without_a_template_an_existing_workbook_is_written_into_rather_than_flattened(
        self,
        tmp_path: Path,
    ) -> None:
        """The decision that keeps ``overwrite_sheets`` meaning one thing: ``path`` is its own template.

        Something has to happen when ``path`` already holds a workbook and no template is named, and
        the only two coherent answers are "write into it" and "destroy it". Destroying it would make
        ``overwrite_sheets`` govern the file after all, in the one case where it is least visible, and it is
        the unrecoverable answer. Writing into it makes the rule exceptionless: this tool never
        removes a tab it was not told about.

        The cost is real and is worth writing down. A flow whose tab names change between runs --
        one tab per month, say -- accumulates tabs in a workbook that nobody ever clears out. The
        fix is a template, which fixes what the pack contains; and the accumulation is visible in
        the file, where the alternative failure is not.
        """
        target = _template(tmp_path / "pack.xlsx", tabs=("Notes",))

        _write(
            {"one": _rows("a")},
            {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures"}]},
        )

        _notes_survived(target)
        assert _tab_names(target) == ["Notes", "Figures"]
        assert _read_back(target, "Figures")["code"].to_list() == ["a", "a", "a"]

    def test_a_replaced_tab_in_the_middle_of_the_pack_comes_back_in_the_middle_of_the_pack(
        self,
        tmp_path: Path,
    ) -> None:
        """Three tabs, and the one replaced is the middle one, so moving it to the end is visible.

        A replaced tab is removed and made again. Where it is made again is the whole question, and
        it can only be asked of a tab with something after it. Replace the last tab of a pack and
        "put it back where it was" and "add it at the end" produce the same workbook, so a writer
        that forgot where the tab was passes.

        What it costs is a person's afternoon. A tab that moved to the end of a pack is a tab the
        reader reports as missing, and the pack opens cleanly, so nothing else says otherwise.
        """
        template = _template(tmp_path / "template.xlsx", tabs=("Notes", "Summary", "Appendix"))
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _rows("a"), "two": _rows("b")},
            {
                "path": str(target),
                "template": str(template),
                "overwrite_sheets": True,
                "sheets": [{"port": "one", "sheet": "Summary"}, {"port": "two", "sheet": "Figures"}],
            },
        )

        assert _tab_names(target) == ["Notes", "Summary", "Appendix", "Figures"], (
            "the replaced tab left the middle of the pack, or the new tab did not go on the end"
        )
        assert _read_back(target, "Summary")["code"].to_list() == ["a", "a", "a"], "the tab was not replaced"
        assert _read_back(target, "Figures")["code"].to_list() == ["b", "b", "b"]
        _notes_survived(target)
        _notes_survived(target, "Appendix")


# ---------------------------------------------------------------------------
# The ports come from the configuration
# ---------------------------------------------------------------------------


class TestThePortsComeFromTheSheets:
    """The first tool in the catalog whose input ports are a property of the step, not the tool."""

    def test_two_steps_with_different_sheets_read_different_ports_in_the_order_written(self) -> None:
        """A static attribute cannot say this, which is the whole reason the member exists.

        Order is part of the answer, not decoration: the ports are reported in the order the author
        wrote them, so a rendered flow lists a step's inputs the way its configuration reads.
        """
        tool = WriteExcel()

        assert ports_from_config(tool) is True
        assert tuple(tool.input_ports) == (), "a tool that derives its ports declares none"
        assert input_ports_for(
            tool,
            {
                "path": "pack.xlsx",
                "sheets": [{"port": "second", "sheet": "B"}, {"port": "first", "sheet": "A"}],
            },
        ) == ("second", "first")
        assert input_ports_for(tool, {"path": "pack.xlsx", "sheets": [{"port": "only", "sheet": "A"}]}) == ("only",)

    def test_the_field_that_names_the_ports_is_one_an_author_always_writes(self) -> None:
        """Derivation reads the configuration as authored, so a defaulted port list would be invisible.

        The loader checks a step's edges before any blank in its configuration is filled in. A
        ``sheets`` field carrying a default would therefore derive nothing at load and something at
        run time, and the two disagreeing is exactly the failure the library has no check for. The
        only defence is that the field is required and carries no default, which is what this pins.
        """
        template = WriteExcel().config_template()

        assert template["sheets"].required is True
        assert template["sheets"].default is None, "a port list behind a default is invisible to the loader"
        assert template["sheets"].non_empty is True, "a workbook with no tabs is not a deliverable"


# ---------------------------------------------------------------------------
# Refusals about the configuration
# ---------------------------------------------------------------------------


class TestRefusalsAboutTheConfiguration:
    """Everything answerable before a workbook is opened, refused before a workbook is opened."""

    def test_two_entries_naming_the_same_tab_are_refused_rather_than_one_overwriting_the_other(
        self,
        tmp_path: Path,
    ) -> None:
        """Two ports, one tab: one port's rows are delivered and the other's are not, with nothing said.

        There is no reading of this configuration that anybody wants. Writing both in order would
        deliver whichever came last and lose the other; merging them is a union, which is a tool
        that already exists and would be silently performed here by a sink. So it is refused, and
        the refusal names the tab rather than the ports, because the tab is what the author has to
        change.
        """
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _rows("a"), "two": _rows("b")},
                {
                    "path": str(target),
                    "sheets": [{"port": "one", "sheet": "Figures"}, {"port": "two", "sheet": "Figures"}],
                },
            )

        message = str(caught.value)
        assert "Figures" in message, f"the refusal does not name the tab written twice: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    def test_two_entries_naming_the_same_port_are_refused_by_the_tool_as_well_as_by_the_loader(
        self,
        tmp_path: Path,
    ) -> None:
        """Refused twice over, deliberately, because the two refusals reach different readers.

        The loader already refuses a step whose derived ports repeat a name, and that is the right
        place for it: the file is wrong every time it is read, whatever is flowing. But the loader
        only speaks when there is a pipeline. A tool run directly -- from a test, from a worked
        example, from anything driving one step -- would otherwise stack one frame onto two tabs and
        report success, because a repeated port name collapses into one entry of the envelope long
        before this tool sees it. Two cheap checks in two places is not duplication when they catch
        the same defect at different moments for different people.
        """
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _rows("a")},
                {
                    "path": str(target),
                    "sheets": [{"port": "one", "sheet": "Figures"}, {"port": "one", "sheet": "Detail"}],
                },
            )

        message = str(caught.value)
        assert "one" in message, f"the refusal does not name the port declared twice: {message}"
        assert "port" in message.lower(), f"the refusal does not say what the defect is: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize(
        ("config", "named"),
        [
            pytest.param({"path": "pack.csv", "sheets": [{"port": "one", "sheet": "A"}]}, ".xlsx", id="not_a_workbook"),
            pytest.param({"path": "pack.xls", "sheets": [{"port": "one", "sheet": "A"}]}, ".xlsx", id="the_older_format"),
            pytest.param({"path": "pack.xlsm", "sheets": [{"port": "one", "sheet": "A"}]}, ".xlsx", id="a_macro_workbook"),
            pytest.param({"path": "pack", "sheets": [{"port": "one", "sheet": "A"}]}, ".xlsx", id="no_extension_at_all"),
            pytest.param({"path": "pack.xlsx", "sheets": []}, "sheets", id="a_workbook_with_no_tabs_at_all"),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "a" * 32}]},
                "31",
                id="a_tab_name_one_character_too_long",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "Q1 [draft]"}]},
                "[",
                id="a_tab_name_holding_a_square_bracket",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "2026/01"}]},
                "/",
                id="a_tab_name_holding_a_slash",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "Assets: current"}]},
                ":",
                id="a_tab_name_holding_a_colon",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "rate * 2"}]},
                "*",
                id="a_tab_name_holding_an_asterisk",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "balance?"}]},
                "?",
                id="a_tab_name_holding_a_question_mark",
            ),
            pytest.param(
                {"path": "pack.xlsx", "sheets": [{"port": "one", "sheet": "2026\\01"}]},
                "\\",
                id="a_tab_name_holding_a_back_slash",
            ),
        ],
    )
    def test_a_configuration_a_workbook_cannot_carry_is_refused_with_the_constraint_named(
        self,
        tmp_path: Path,
        config: dict[str, Any],
        named: str,
    ) -> None:
        """Refused by this library, in its own words, rather than by the writer underneath.

        A tab name is written by a person for a person, so it attracts exactly the characters a
        workbook forbids: a date with slashes, a heading with a colon, a draft marker in brackets.
        Left to the library underneath, a whole pack fails partway through with a message about
        worksheets; left to a writer that silently renames, the pack is delivered with a tab nobody
        can find under the name the flow calls it.
        """
        target = tmp_path / str(config["path"])
        config = {**config, "path": str(target)}
        # The frames are built from the configuration rather than written out, because the ports
        # this step reads *are* the configuration: a case naming no tabs reads nothing.
        frames = {str(entry["port"]): _rows("a") for entry in config["sheets"]}

        with pytest.raises(ConfigError) as caught:
            _write(frames, config)

        assert named in str(caught.value), f"the refusal does not name the constraint broken: {caught.value}"
        assert not target.exists(), "nothing is created by a refusal"

    def test_the_longest_tab_name_allowed_is_accepted(self, tmp_path: Path) -> None:
        """The mirror of the too-long case: 31 characters is a name, so the check is not one too tight."""
        target = tmp_path / "pack.xlsx"

        _write({"one": _rows("a")}, {"path": str(target), "sheets": [{"port": "one", "sheet": "a" * 31}]})

        assert _tab_names(target) == ["a" * 31]


# ---------------------------------------------------------------------------
# Refusals about the frames, carried over whole from the single-sheet sink
# ---------------------------------------------------------------------------


class TestRefusalsAboutTheFrames:
    """The destination is the same destination, so the rulings and the numbers are the same."""

    @pytest.mark.parametrize(
        ("value", "outcome"),
        [
            pytest.param(MAX_EXACT_INTEGER + 1, "refused", id="one_past_the_largest_exact_whole_number"),
            pytest.param(123456789012345678, "refused", id="an_identifier_of_the_length_a_real_one_has"),
            pytest.param(-(MAX_EXACT_INTEGER + 1), "refused", id="and_the_same_going_the_other_way"),
            pytest.param(MAX_EXACT_INTEGER, "written", id="but_the_largest_exact_whole_number_is_written"),
        ],
    )
    def test_a_whole_number_a_cell_cannot_keep_exactly_stops_the_run_as_a_data_fault(
        self,
        tmp_path: Path,
        value: int,
        outcome: str,
    ) -> None:
        """A cell holds a double, so this value cannot come back as itself, and neither alternative is honest.

        123456789012345678 reads back 123456789012345696, and an identifier whose last digits
        changed matches nothing, joins to nothing and reconciles to nothing, with no error anywhere.
        Writing it as text instead changes the column's type without being asked. Letting it through
        is the corruption itself. So it is refused, the rows are named, and the fix is one explicit
        cast upstream where the next reader can see the decision.

        The offending column is on the second port, because a writer that checks the first frame and
        then starts writing has a whole tab on disk before it finds this one.
        """
        target = tmp_path / "pack.xlsx"
        frames = {"one": _rows("a"), "two": pl.DataFrame({"key": ["r1", "r2"], "reference": [1, value]})}
        config = {
            "path": str(target),
            "sheets": [{"port": "one", "sheet": "Figures"}, {"port": "two", "sheet": "Detail"}],
        }

        if outcome == "written":
            _write(frames, config)
            assert _read_back(target, "Detail")["reference"].to_list() == [1, value]
            return

        with pytest.raises(DataError) as caught:
            _write(frames, config)

        message = str(caught.value)
        assert "reference" in message, "which column holds it"
        assert str(value) in message, "and the value itself, so it can be looked up"
        assert not target.exists(), "a refused write creates nothing, not even the tabs it got to first"

    @pytest.mark.parametrize(
        "column",
        [
            pytest.param(
                pl.Series("amount", [Decimal("1234.56")], dtype=pl.Decimal(12, 2)),
                id="a_figure_whose_whole_purpose_is_exactness",
            ),
            pytest.param(pl.Series("at", [dt.time(13, 45)], dtype=pl.Time), id="a_time_of_day_lands_in_1899"),
            pytest.param(
                pl.Series("took", [dt.timedelta(hours=2)], dtype=pl.Duration),
                id="a_length_of_time_becomes_a_fraction_of_a_day",
            ),
            pytest.param(
                pl.Series("parts", [[1, 2]], dtype=pl.List(pl.Int64)),
                id="a_list_becomes_a_line_of_punctuation",
            ),
            pytest.param(
                pl.Series("detail", [{"x": 1}], dtype=pl.Struct({"x": pl.Int64})),
                id="and_so_does_a_nested_record",
            ),
            pytest.param(
                pl.Series("parts", [[1, 2]], dtype=pl.Array(pl.Int64, 2)),
                id="a_fixed_size_list_is_a_list_too",
            ),
            pytest.param(
                pl.Series("blob", [b"0064"], dtype=pl.Binary),
                id="raw_bytes_are_quietly_read_as_text",
            ),
            pytest.param(
                pl.Series(
                    "at",
                    [dt.datetime(2026, 1, 31, 13, 45, tzinfo=dt.UTC)],
                    dtype=pl.Datetime("us", "UTC"),
                ),
                id="an_instant_that_knows_its_zone_arrives_without_one",
            ),
        ],
    )
    def test_a_column_a_cell_cannot_hold_is_refused_as_a_fault_in_the_shape(
        self,
        tmp_path: Path,
        column: pl.Series,
    ) -> None:
        """Each of these has a plausible-looking wrong answer in a cell, and none is the value it was.

        Reported as a configuration fault rather than a data fault: no row is more at fault than any
        other, there is nothing to show a reader in the rows, and the fix is authoring -- a step
        upstream that formats the column the way the author wants it to appear.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame({"key": ["r1"]}).with_columns(column)

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _rows("a"), "two": frame},
                {
                    "path": str(target),
                    "sheets": [{"port": "one", "sheet": "Figures"}, {"port": "two", "sheet": "Detail"}],
                },
            )

        message = str(caught.value)
        assert column.name in message, "which column to go and fix"
        assert str(column.dtype) in message, "and what it currently holds"
        assert not target.exists(), "a refused write creates nothing"

    def test_a_frame_taller_than_a_tab_is_refused_by_name_rather_than_quietly_truncated(
        self,
        tmp_path: Path,
    ) -> None:
        """Truncation is the worst outcome a delivered figure can have, so the limit refuses instead.

        The pack opens, every figure in it is plausible, and the total is wrong by however many rows
        fell off the end. The check reads the frame's height, so it costs nothing and happens before
        anything is opened -- which is also how this pins the rule without writing a million rows.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.select(pl.int_range(0, MAX_DATA_ROWS + 1, dtype=pl.Int64).alias("key"))

        with pytest.raises(DataError) as caught:
            _write(
                {"one": frame},
                {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures"}]},
            )

        message = str(caught.value)
        assert str(MAX_DATA_ROWS) in message or f"{MAX_DATA_ROWS:,}" in message, "the refusal names the limit"
        assert not target.exists(), "nothing was written on the way to finding out"

    def test_a_refused_write_leaves_the_template_and_an_existing_pack_exactly_as_it_found_them(
        self,
        tmp_path: Path,
    ) -> None:
        """Refusing late is not refusing: a half-written pack is worse than no pack at all.

        The case with the most to lose is picked on purpose -- an authorised replacement of a tab in
        a pack that already exists, which a writer that opened and saved as it went would have
        already destroyed by the time it noticed the second frame was unwritable.
        """
        template = _template(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"
        target.write_bytes(b"last month's pack")
        frame = pl.DataFrame({"key": ["r1"], "took": [dt.timedelta(hours=2)]})

        with pytest.raises(ConfigError):
            _write(
                {"one": _rows("a"), "two": frame},
                {
                    "path": str(target),
                    "template": str(template),
                    "overwrite_sheets": True,
                    "sheets": [{"port": "one", "sheet": "Summary"}, {"port": "two", "sheet": "Detail"}],
                },
            )

        assert target.read_bytes() == b"last month's pack"
        _notes_survived(template)
        assert sorted(item.name for item in tmp_path.iterdir()) == ["pack.xlsx", "template.xlsx"], (
            "a refused write left a scratch file behind"
        )


# ---------------------------------------------------------------------------
# One load and one save
# ---------------------------------------------------------------------------


class TestOneLoadAndOneSave:
    """The reason this is one tool rather than one step per tab, pinned rather than asserted in prose."""

    def test_a_pack_of_several_tabs_is_saved_once(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Four tabs, one save. A writer that saved per tab would also have loaded per tab.

        This is the one test here coupled to the library underneath rather than to behaviour, and it
        is worth the coupling. Saving once is the property the whole design rests on: every extra
        save is another chance for the curated parts of the workbook to be rewritten, and a pack of
        eleven tabs saved eleven times costs eleven full serialisations of a file that can be
        megabytes. The counter is on the class rather than the module function, so it counts a save
        however the tool spells its import.
        """
        template = _template(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"
        tabs = ("First", "Second", "Third", "Fourth")

        # Counted only from here, so the template this test built itself is not one of the saves.
        saves: list[str] = []
        original = openpyxl.workbook.workbook.Workbook.save

        def counting_save(self: Any, filename: Any) -> Any:
            saves.append(str(filename))
            return original(self, filename)

        monkeypatch.setattr(openpyxl.workbook.workbook.Workbook, "save", counting_save)

        _write(
            {f"p{index}": _rows(str(index)) for index in range(len(tabs))},
            {
                "path": str(target),
                "template": str(template),
                "sheets": [{"port": f"p{index}", "sheet": tab} for index, tab in enumerate(tabs)],
            },
        )

        assert saves == [str(target)], f"the pack was saved {len(saves)} time(s), and never to anywhere else"
        assert _tab_names(target) == ["Notes", "Summary", *tabs]

    def test_a_pack_is_loaded_once_as_well_as_saved_once_and_the_delivered_tabs_are_the_ones_written(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One load, one save, and the pack on disk is the book the tabs went into.

        Counting saves alone leaves two real mutations unnoticed, and they are not hypothetical.

        A "which tabs are already in there?" pre-check that opens the workbook a second time still
        saves once. Nothing here would say a word, and the cost is the whole point of the tool: a run
        that took six and a half minutes takes thirteen, with nobody told why.

        Worse: open the workbook, write every tab into it, then save a second book loaded separately.
        One save, correct tab names, and every figure written is absent from the delivered pack. So
        the contents of a tab are read back here, not just the names -- names alone would let that
        through.

        Zero loads is pinned as well, on a run with no template and nothing at the path. There is
        nothing to open, so opening anything is a defect.
        """
        template = _template(tmp_path / "template.xlsx")
        target = tmp_path / "pack.xlsx"
        markers = ("a", "b", "c", "d")
        tabs = ("First", "Second", "Third", "Fourth")

        # Counted only from here, so the template this test built itself is not one of them.
        loads: list[str] = []
        saves: list[str] = []
        original_load = openpyxl.load_workbook
        original_save = openpyxl.workbook.workbook.Workbook.save

        def counting_load(filename: Any, *args: Any, **kwargs: Any) -> Any:
            loads.append(str(filename))
            return original_load(filename, *args, **kwargs)

        def counting_save(self: Any, filename: Any) -> Any:
            saves.append(str(filename))
            return original_save(self, filename)

        monkeypatch.setattr(openpyxl, "load_workbook", counting_load)
        monkeypatch.setattr(openpyxl.workbook.workbook.Workbook, "save", counting_save)

        _write(
            {f"p{marker}": _rows(marker) for marker in markers},
            {
                "path": str(target),
                "template": str(template),
                "sheets": [{"port": f"p{marker}", "sheet": tab} for marker, tab in zip(markers, tabs, strict=True)],
            },
        )
        from_a_template = (list(loads), list(saves))

        loads.clear()
        saves.clear()
        plain = tmp_path / "plain.xlsx"
        _write({"one": _rows("a")}, {"path": str(plain), "sheets": [{"port": "one", "sheet": "Figures"}]})
        from_nothing = (list(loads), list(saves))

        monkeypatch.undo()

        assert from_a_template == ([str(template)], [str(target)]), (
            f"a run from a template opened {len(from_a_template[0])} workbook(s) and saved "
            f"{len(from_a_template[1])}; it opens the template once and saves the pack once"
        )
        assert from_nothing == ([], [str(plain)]), (
            f"a run with no template and no pack yet opened {len(from_nothing[0])} workbook(s); "
            f"there is nothing to open"
        )
        assert _tab_names(target) == ["Notes", "Summary", *tabs]
        assert _read_back(target, "Third")["code"].to_list() == ["c", "c", "c"], (
            "the delivered pack is not the workbook the tabs were written into"
        )
        assert _read_back(plain, "Figures")["code"].to_list() == ["a", "a", "a"]


# ---------------------------------------------------------------------------
# The template and the catalog
# ---------------------------------------------------------------------------


class TestWriteExcelInTheCatalog:
    """Described, documented with examples that actually run, terminal, and resolvable by name."""

    def test_the_template_says_where_which_tabs_and_what_may_be_overwritten(self) -> None:
        """Four fields, and the shape of ``sheets`` is declared rather than left as an untyped array.

        The inner shape is load-bearing twice over. It is what type-checks a sheet entry an author
        wrote, and it is what a reader of the catalog page needs in order to write one at all: a
        bare ``array`` field describes a list of anythings. ``path`` is a plain string and
        emphatically not an expression, because a loader pre-flighting the flow would otherwise try
        to compile a filename as a column reference.
        """
        template = WriteExcel().config_template()

        assert set(template) == {"path", "sheets", "template", "overwrite_sheets"}
        assert template["path"].type == "string"
        assert template["path"].required is True
        assert template["path"].non_empty is True, "an empty path is a value, and not one to act on"
        assert template["sheets"].type == "array"
        assert template["template"].type == "string"
        assert template["template"].required is False, "most packs start from nothing"
        assert template["overwrite_sheets"].type == "boolean"
        assert template["overwrite_sheets"].default is False, "a tab that is already there survives until an author says so"

        entry = template["sheets"].items
        assert entry is not None and entry.type == "object", "a sheet entry has a declared shape"
        assert entry.fields is not None
        # 'formats' is the third member, and its own shape is pinned where it is argued, below.
        assert set(entry.fields) == {"port", "sheet", "formats", "widths", "fit_widths", "wrap", "table", "header"}
        assert entry.fields["port"].required is True and entry.fields["port"].non_empty is True
        assert entry.fields["sheet"].required is True and entry.fields["sheet"].non_empty is True
        assert expression_fields(template) == (), "nothing here is a column reference"

    def test_write_excel_is_terminal_and_conforms_twice_over_leaving_nothing_behind(self) -> None:
        """A sink with no output ports at all, and worked examples that really write and really clean up.

        Terminality is the decision here: handing eleven frames back out is a fan-out this tool
        would have invented, and handing back a receipt is shape without information. So it
        declares no output ports, and a flow ends at it.

        The check runs twice, because an example that declares the world it needs -- a template
        already on disk -- declares it as a callable returning a context manager precisely so it can
        be entered again. A setup good for one run passes the first check and fails the second,
        which is the shape of defect that only shows up for whoever runs the catalog check after
        you. Nothing may be left in the working directory either: an example writing to a relative
        path drops a workbook wherever the check was run from.
        """
        tool = WriteExcel()
        before = sorted(Path.cwd().iterdir())

        check_tool(tool)
        check_tool(tool)

        assert sorted(Path.cwd().iterdir()) == before, "the catalog check littered the working directory"
        assert tool.name == "write_excel"
        assert tool.category == "output"
        assert tool.category in CATEGORIES
        assert tuple(tool.output_ports) == (), "a workbook writer has nothing honest to hand on"

        help_ = tool.help()
        assert len(help_.examples) >= 2, "one plain pack, and one written into a template"
        for example in help_.examples:
            assert str(example.configuration["path"]).startswith("{scratch}/"), (
                f"a reader has to be able to copy this straight off the page: {example.configuration['path']}"
            )
            assert example.outputs.names == (), "a terminal tool's example documents no output"
        assert any("template" in example.configuration for example in help_.examples), (
            "the case the tool exists for has to be the case it demonstrates"
        )

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "template" in documented
        assert "overwrite_sheets" in documented, "the destructive case has to be documented"
        assert "exact" in documented or "precision" in documented, "so does what a cell cannot keep"

    def test_write_excel_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "write_excel" in REGISTRY
        assert isinstance(REGISTRY.get("write_excel"), WriteExcel)


# ---------------------------------------------------------------------------
# The pivot tables in a delivered pack
# ---------------------------------------------------------------------------


def _pack_refreshed_from_the_pivot_template(tmp_path: Path) -> Path:
    """The committed pivot workbook used as a template, with one data tab written into it."""
    target = tmp_path / "pack.xlsx"
    _write(
        {"one": _rows("a")},
        {
            "path": str(target),
            "template": str(PIVOT_TEMPLATE),
            "sheets": [{"port": "one", "sheet": "Figures"}],
        },
    )
    return target


class TestThePacksPivotTablesSurvive:
    """The claim the tool is bought for, measured on a workbook that really has a pivot in it.

    Everything else here reads the pack through a spreadsheet library. These read the zip, because
    what is under test is the parts of the file rather than the values in the cells. No spreadsheet
    program is involved, so these run anywhere.
    """

    def test_every_pivot_part_of_the_template_has_a_counterpart_in_the_pack(self, tmp_path: Path) -> None:
        """A pivot is four parts and a web of relationships, and losing any one of them loses the pivot.

        Parts are counted rather than named. A writer may renumber one -- pivotCacheDefinition1
        becoming pivotCacheDefinition2 -- and it has lost nothing by doing it, so a test matching on
        names would report a failure that is not one.

        The manifest is checked beside the parts because the two fail apart. A part that survives in
        the zip but loses its content type is a part nothing will open, and the pack arrives as a
        workbook a reader is asked to repair -- which takes the pivot out.
        """
        target = _pack_refreshed_from_the_pivot_template(tmp_path)

        assert _pivot_parts(target) == _pivot_parts(PIVOT_TEMPLATE), "the pack is missing a pivot part"
        assert _pivot_content_types(target) == _pivot_content_types(PIVOT_TEMPLATE), (
            "a pivot part survived but stopped being declared, so nothing will open it"
        )
        assert _dangling_relationships(target) == [], "a relationship points at a part that is not there"
        assert _tab_names(target) == ["Data", "Pivot", "Figures"]
        assert _read_back(target, "Figures")["code"].to_list() == ["a", "a", "a"]

    def test_each_pivot_cache_still_holds_the_values_that_fill_its_filter_dropdown(self, tmp_path: Path) -> None:
        """A cache that keeps its claim and drops its values delivers a pivot with empty filters.

        The cached values under ``sharedItems`` are what a reader sees in a pivot's filter list
        before anybody presses refresh. They are also invisible: the pivot draws, the totals are
        right, and only the dropdown is empty. So the count of them is pinned beside the record count
        the cache declares, because the two disagreeing is the shape of the damage.
        """
        target = _pack_refreshed_from_the_pivot_template(tmp_path)

        written = _cache_definitions(target)
        original = _cache_definitions(PIVOT_TEMPLATE)

        assert len(written) == len(original)
        for after, before in zip(written, original, strict=True):
            assert after["record_count"] == before["record_count"], "the cache changed how many rows it claims"
            assert after["shared_item_blocks"] == before["shared_item_blocks"], "a whole shared-items block is gone"
            assert after["shared_item_values"] == before["shared_item_values"], (
                "the cached values are gone; the pivot's filter dropdowns arrive empty"
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Not true today, and recorded rather than hidden. The pack is written by loading the "
            "template into an object model and serialising it back out, and that model has no place "
            "for the 'ext' blocks on a pivot cache -- which is where a spreadsheet program keeps the "
            "parts of a pivot that postdate the original file format, a slicer link among them. "
            "Measured on this fixture: the one cache definition goes in at 1,141 bytes and comes out "
            "at 921, with its single 'ext' block gone. The same rewrite on a real 36 MB pack with 27 "
            "pivots took one cache definition from 6,951 bytes to 2,645. Every part survives and no "
            "relationship dangles, so nothing else here notices. Until this passes, the tool's own "
            "prose must not tell a reader the workbook keeps its own parts intact."
        ),
    )
    def test_a_pivot_caches_extension_block_survives_being_written_back_out(self, tmp_path: Path) -> None:
        """The measured loss, pinned so the day it stops happening we are told rather than left guessing."""
        target = _pack_refreshed_from_the_pivot_template(tmp_path)

        written = _cache_definitions(target)
        original = _cache_definitions(PIVOT_TEMPLATE)

        for after, before in zip(written, original, strict=True):
            assert after["extensions"] == before["extensions"], (
                f"the cache definition lost {before['extensions'] - after['extensions']} extension "
                f"block(s), and shrank from {before['bytes']} bytes to {after['bytes']}"
            )


# ---------------------------------------------------------------------------
# How a column's numbers are shown
# ---------------------------------------------------------------------------

MONEY_FORMAT = "#,##0.00"
"""The format every money column of a delivered pack carries: grouped thousands, two decimals."""

DATE_FORMAT = "yyyy-mm-dd"
"""The format every date column carries, and the one this tool already writes on its own."""

THREE_DECIMALS = "0.000"
"""The one format in a measured pack that belongs to a single column rather than to a type."""

DATETIME_FORMAT = "yyyy-mm-dd hh:mm:ss"
"""The format every instant column carries before any rule is read, hours declared as ``hh``."""

BRITISH_DATE_FORMAT = "dd/mm/yyyy"
"""A date shown the way a reader here writes one, and deliberately not the format above.

A test asserting a Date rule produces ``yyyy-mm-dd`` cannot fail, because that is what the column
would carry under no rule at all. Every date override below is stated in a format the default is
not, so the assertion is about the rule rather than about the tool's own habit.
"""

BRITISH_DATETIME_FORMAT = "dd/mm/yyyy hh:mm"
"""An instant shown to the minute, and again not what the tool writes on its own."""

LONG_DATE_FORMAT = "d mmm yyyy"
"""A third spelling of a date, for the rule that names one column rather than the type."""

WHOLE_NUMBER_FORMAT = "#,##0"
"""A count with grouped thousands and no decimal part: what a rule about Int64 is for."""

TEXT_FORMAT = "@"
"""The code that says 'show this as text', and the only thing a rule about String has to say."""

YAML_REWRITES = frozenset({"on", "off", "yes", "no", "true", "false", "null", "y", "n"})
"""Bare words a definition file hands over as something other than the text written."""

A_LONG_FIGURE = 661617.4665392789
"""A figure with more decimals than any format shows, so rounding on the way in is visible."""


def _formatted_rows(height: int = 3) -> pl.DataFrame:
    """One frame carrying every type these rules speak about, plus one column that overrides.

    ``height`` is here for the refusals: a configuration question has to be answered the same way
    whether three rows arrived or none, and a frame with no rows still has a schema.
    """
    frame = pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "amount": [A_LONG_FIGURE, -5.5, 1234.56],
            "balance": [0.125, 2.0, -99.999],
            "Variance %": [0.5, 0.25, 0.125],
            "count": [1, 2, 3],
            "as_at": [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 3, 31)],
        }
    )
    return frame.head(height)


def _dated_rows() -> pl.DataFrame:
    """A date column and an instant column side by side, which is what a date override needs.

    Kept apart from :func:`_formatted_rows` rather than bolted onto it. The two temporal types
    default to different formats and a rule about one must leave the other alone, so every case
    below needs both columns present and needs to be able to say which of the two moved.
    """
    return pl.DataFrame(
        {
            "key": ["r1", "r2", "r3"],
            "as_at": [dt.date(2026, 1, 31), dt.date(2026, 2, 28), dt.date(2026, 3, 31)],
            "recorded_at": [
                dt.datetime(2026, 1, 31, 13, 45, 30),
                dt.datetime(2026, 2, 28, 0, 0, 0),
                dt.datetime(2026, 3, 31, 23, 59, 59),
            ],
        }
    )


def _formats(path: Path, sheet: str) -> dict[str, list[str]]:
    """Every data cell's number format, under the header of the column it sits in.

    The format code is read off the written file rather than taken from the configuration, because
    the configuration is the claim and the file is the delivery. A tool that recorded the rules and
    never applied them would satisfy any assertion made against the config and none made here.
    """
    book = openpyxl.load_workbook(path)
    try:
        worksheet = book[sheet]
        headers = [cell.value for cell in worksheet[1]]
        rows = range(2, worksheet.max_row + 1)
        return {
            str(header): [worksheet.cell(row=row, column=index).number_format for row in rows]
            for index, header in enumerate(headers, start=1)
        }
    finally:
        book.close()


def _header_formats(path: Path, sheet: str) -> list[str]:
    """The number format on each cell of the header row."""
    book = openpyxl.load_workbook(path)
    try:
        return [cell.number_format for cell in book[sheet][1]]
    finally:
        book.close()


def _raw_values(path: Path, sheet: str, column: str) -> list[Any]:
    """One column's values as the cells hold them, with nothing inferred on the way back."""
    book = openpyxl.load_workbook(path)
    try:
        worksheet = book[sheet]
        index = [cell.value for cell in worksheet[1]].index(column) + 1
        return [worksheet.cell(row=row, column=index).value for row in range(2, worksheet.max_row + 1)]
    finally:
        book.close()


class TestAColumnsNumbersAreShownAsTheSheetSays:
    """``formats`` says how a column's figures are shown, and says nothing about what they are.

    The pack this replaces opens with every money column reading ``661617.4665392789`` where the
    delivered one reads ``661,617.47``. Both files hold the same number. The difference is one
    number format per column, and without a way to state it the figures are right and the file is
    not presentable.

    Two shapes of rule, because two questions are being asked. A ``type`` rule reaches every column
    of a type, including the ones nobody could have named: a pivot tab gains a column per period,
    so naming them individually means editing the pipeline file every month. A ``column`` rule names
    one column and beats the type rule, which is how one column reads to three decimals while every
    other number on the same tab reads as money.

    Nothing is guessed. A column no rule matched keeps the workbook's default, because a tool that
    decided a Float64 was money would be wrong the first time somebody wrote a reference number in
    one. The engine this replaces guessed from the column's *name* -- a Float64 is money unless the
    name holds a percent sign -- which breaks silently the day a column is renamed.
    """

    def test_a_type_rule_reaches_every_column_of_that_type_and_no_other_column(self, tmp_path: Path) -> None:
        """One rule, two money columns formatted, and the text, the whole number and the header left alone.

        Every data row is checked, not the first: a writer that formats as it lays down row two and
        forgets afterwards delivers a tab that looks right in a screenshot and wrong in the file.
        The header is checked in the same breath, because a heading is a word and a money format on
        a word shows nothing while saying the tool cannot tell a header from a figure.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [{"type": "Float64", "format": MONEY_FORMAT}],
                    }
                ],
            },
        )

        formats = _formats(target, "Figures")
        for column in ("amount", "balance", "Variance %"):
            assert formats[column] == [MONEY_FORMAT] * 3, f"{column} is a Float64 the type rule did not reach"
        assert formats["key"] == ["General"] * 3, "a text column was formatted by a rule about numbers"
        assert formats["count"] == ["General"] * 3, "a whole number was formatted by a rule about decimals"
        assert _header_formats(target, "Figures") == ["General"] * 6, "the header row was formatted like the data"

    @pytest.mark.parametrize(
        "rules",
        [
            pytest.param(
                [
                    {"type": "Float64", "format": MONEY_FORMAT},
                    {"type": "Date", "format": DATE_FORMAT},
                    {"column": "Variance %", "format": THREE_DECIMALS},
                ],
                id="the_general_rule_written_first",
            ),
            pytest.param(
                [
                    {"column": "Variance %", "format": THREE_DECIMALS},
                    {"type": "Float64", "format": MONEY_FORMAT},
                    {"type": "Date", "format": DATE_FORMAT},
                ],
                id="the_exception_written_first",
            ),
        ],
    )
    def test_a_column_rule_beats_the_type_rule_whichever_order_the_two_were_written_in(
        self,
        tmp_path: Path,
        rules: list[dict[str, Any]],
    ) -> None:
        """The exception is written down, its neighbours of the same type are untouched, and the
        order the author happened to write the two rules in does not decide which one wins.

        This is the whole reason a ``column`` rule exists. Three Float64 columns sit side by side;
        two are money and one is a proportion read to three decimals.

        Both orders are asked because only one of them can catch a last-rule-wins implementation,
        and it is not the one a person writes by hand. Stating the general rule first and the
        exception under it reads most naturally and is also the order in which resolving by position
        gives the right answer by accident. Reverse the two -- an author who wrote the odd column
        down while it was in mind, then swept up the rest -- and a writer taking the last matching
        rule shows a percentage column as money and says nothing. Precedence here is about which
        kind of rule it is, never about where in the list it sits, so the same assertions have to
        hold for both orders and are written once so the two cannot drift apart.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [{"port": "one", "sheet": "Figures", "formats": rules}],
            },
        )

        formats = _formats(target, "Figures")
        assert formats["Variance %"] == [THREE_DECIMALS] * 3, "the named column did not beat the rule about its type"
        assert formats["amount"] == [MONEY_FORMAT] * 3, "a neighbour of the same type lost its type rule"
        assert formats["balance"] == [MONEY_FORMAT] * 3, "a neighbour of the same type lost its type rule"
        assert formats["as_at"] == [DATE_FORMAT] * 3

    def test_a_column_no_rule_matched_keeps_the_workbooks_own_default(self, tmp_path: Path) -> None:
        """Formatting is opt-in, so a tab that states one rule states exactly one rule.

        A tool that saw a ``formats`` list and started deciding for itself what the rest of the
        columns ought to look like would be making the claim this design refuses to make. The date
        column is the one exception, and it is not a claim about presentation: without a date format
        the cell holds a count of days since 1899 and reads as one.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [{"column": "amount", "format": MONEY_FORMAT}],
                    }
                ],
            },
        )

        formats = _formats(target, "Figures")
        assert formats["amount"] == [MONEY_FORMAT] * 3
        for column in ("key", "balance", "Variance %", "count"):
            assert formats[column] == ["General"] * 3, f"{column} was given a format nobody asked for"
        assert formats["as_at"] == [DATE_FORMAT] * 3, "the date format is not decoration and does not become optional"

    def test_the_figure_stored_in_the_cell_is_the_figure_that_went_in(self, tmp_path: Path) -> None:
        """The one that matters: a format changes how a number is shown and never what it is.

        ``661617.4665392789`` shown under ``#,##0.00`` reads ``661,617.47``, and the cell still
        holds every digit. A writer that rounded, truncated or wrote the shown text into the cell
        would pass every format assertion above and would have restated a delivered figure. Read
        through the raw cells rather than a reader, because a reader hands back what it parsed and
        this is a question about what was stored.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [
                            {"type": "Float64", "format": MONEY_FORMAT},
                            {"column": "Variance %", "format": THREE_DECIMALS},
                        ],
                    }
                ],
            },
        )

        stored = _raw_values(target, "Figures", "amount")
        assert stored == [A_LONG_FIGURE, -5.5, 1234.56], "a format changed the figures it was asked to show"
        assert repr(stored[0]) == repr(A_LONG_FIGURE), f"the figure was rewritten to {stored[0]!r}"
        assert _raw_values(target, "Figures", "Variance %") == [0.5, 0.25, 0.125]
        assert _read_back(target, "Figures")["amount"].to_list() == [A_LONG_FIGURE, -5.5, 1234.56]

    def test_a_tab_that_states_no_formats_is_written_exactly_as_it_was_before(self, tmp_path: Path) -> None:
        """The field is new, so a pipeline that never heard of it must be written the old way.

        Dates still carry the format the tool has always written, and nothing else carries anything.
        A tool that started formatting numbers on its own would change every pack already shipping.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures"}]},
        )

        formats = _formats(target, "Figures")
        assert formats["as_at"] == [DATE_FORMAT] * 3, "a date lost the one format this tool has always written"
        for column in ("key", "amount", "balance", "Variance %", "count"):
            assert formats[column] == ["General"] * 3, f"{column} was formatted by a tab that asked for nothing"

    def test_a_type_rule_reaches_columns_whose_names_the_configuration_could_not_have_known(
        self,
        tmp_path: Path,
    ) -> None:
        """The reason a rule may speak about a type at all: a pivot tab gains a column every period.

        These names come out of the data. Naming them one by one means editing the pipeline file
        every month, which is the coupling the rest of this pipeline was built to avoid -- and the
        month somebody forgets, the new column is the only unformatted one on the tab.
        """
        target = tmp_path / "pack.xlsx"
        # Built here the way a cross-tab builds them: from the rows, long after the config was written.
        periods = {f"2025-{month:02d}": [float(month) * 1000.5, float(month)] for month in range(1, 6)}
        frame = pl.DataFrame({"key": ["r1", "r2"], **periods})

        _write(
            {"one": frame},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Pivot",
                        "formats": [{"type": "Float64", "format": MONEY_FORMAT}],
                    }
                ],
            },
        )

        formats = _formats(target, "Pivot")
        assert set(formats) == {"key", *periods}, "the tab did not carry the columns the frame did"
        for period in periods:
            assert formats[period] == [MONEY_FORMAT] * 2, f"{period} arrived unformatted"

    def test_the_rules_of_one_tab_do_not_reach_another_tab_of_the_same_pack(self, tmp_path: Path) -> None:
        """One workbook, three tabs, three different answers -- including a tab that asked for nothing.

        Formats are declared per sheet entry, so a pack that formats its summary as money and leaves
        its detail alone must get exactly that. A writer holding one set of rules across the loop
        delivers the last tab's rules everywhere, which reads as correct on any single-tab test.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows(), "two": _formatted_rows(), "three": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Money",
                        "formats": [{"type": "Float64", "format": MONEY_FORMAT}],
                    },
                    {
                        "port": "two",
                        "sheet": "Proportions",
                        "formats": [{"column": "Variance %", "format": THREE_DECIMALS}],
                    },
                    {"port": "three", "sheet": "Plain"},
                ],
            },
        )

        money = _formats(target, "Money")
        proportions = _formats(target, "Proportions")
        plain = _formats(target, "Plain")

        assert money["amount"] == [MONEY_FORMAT] * 3
        assert money["Variance %"] == [MONEY_FORMAT] * 3, "another tab's column rule reached this one"
        assert proportions["Variance %"] == [THREE_DECIMALS] * 3
        assert proportions["amount"] == ["General"] * 3, "another tab's type rule reached this one"
        assert plain["amount"] == ["General"] * 3, "a tab that stated no rules was given another tab's"
        assert plain["Variance %"] == ["General"] * 3

    @pytest.mark.parametrize(
        ("rules", "on_the_date", "on_the_instant"),
        [
            pytest.param(
                [{"type": "Date", "format": BRITISH_DATE_FORMAT}],
                BRITISH_DATE_FORMAT,
                DATETIME_FORMAT,
                id="a_rule_about_the_date_type",
            ),
            pytest.param(
                [{"type": "Datetime", "format": BRITISH_DATETIME_FORMAT}],
                DATE_FORMAT,
                BRITISH_DATETIME_FORMAT,
                id="a_rule_about_the_instant_type",
            ),
            pytest.param(
                [{"column": "as_at", "format": LONG_DATE_FORMAT}],
                LONG_DATE_FORMAT,
                DATETIME_FORMAT,
                id="a_rule_naming_one_date_column",
            ),
        ],
    )
    def test_a_rule_about_a_date_replaces_the_format_this_tool_writes_on_its_own(
        self,
        tmp_path: Path,
        rules: list[dict[str, Any]],
        on_the_date: str,
        on_the_instant: str,
    ) -> None:
        """The date format is a default the pipeline may overrule, not a guarantee it may not.

        Every format here is deliberately one the tool would never have written by itself. A case
        stating ``yyyy-mm-dd`` for a Date column asserts nothing at all, because that is exactly what
        the column carries under no rule: the tool could ignore the rule outright, or give the
        built-in the last word, and the assertion would still pass. Asking for ``dd/mm/yyyy`` is the
        only way to see which of the two reached the cell.

        Three cases because there are three separate paths in. A rule about ``Date``, a rule about
        ``Datetime`` -- whose default differs and must not be moved by the other's rule -- and a rule
        naming one date column by name, which resolves through the by-column branch and would
        otherwise never be entered by any test in this file.

        The column left un-ruled in each case is checked in the same breath. A writer that let any
        temporal rule replace both defaults would satisfy half these assertions and deliver a pack
        whose timestamps lost their time.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _dated_rows()},
            {
                "path": str(target),
                "sheets": [{"port": "one", "sheet": "Dates", "formats": rules}],
            },
        )

        formats = _formats(target, "Dates")
        assert formats["as_at"] == [on_the_date] * 3, "the date column is not shown the way the tab asked"
        assert formats["recorded_at"] == [on_the_instant] * 3, "the instant column is not shown the way the tab asked"
        assert formats["key"] == ["General"] * 3, "a text column was reached by a rule about dates"
        assert _read_back(target, "Dates")["as_at"].to_list() == _dated_rows()["as_at"].to_list(), (
            "the format a column is shown under changed the dates it was asked to show"
        )

    def test_a_cell_with_nothing_in_it_is_formatted_like_the_rest_of_its_column(self, tmp_path: Path) -> None:
        """A money column with a gap in it is still a money column, all the way down.

        This is the defect the whole field exists to remove, in its least obvious form. Formatting
        only the cells that happened to hold a value leaves a chequerboard: the figures are right,
        the file is not presentable, and the gaps announce themselves the moment somebody types into
        one and it comes out unformatted beside its neighbours. The format belongs to the cell, not
        to the value that happens to be sitting in it.

        The empty string is here in the same breath because the writer rewrites one to a blank cell
        on the way in. That rewrite is the second way a cell ends up holding nothing, and a writer
        formatting on the value rather than on the column would drop the format on exactly those
        cells too -- in a text column, where a missing ``@`` is invisible until a code like ``007``
        arrives and is shown as ``7``.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame(
            {
                "key": ["r1", "r2", "r3"],
                "amount": [A_LONG_FIGURE, None, 1234.56],
                "code": ["A", "", "C"],
            }
        )

        _write(
            {"one": frame},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [
                            {"type": "Float64", "format": MONEY_FORMAT},
                            {"column": "code", "format": TEXT_FORMAT},
                        ],
                    }
                ],
            },
        )

        formats = _formats(target, "Figures")
        assert formats["amount"] == [MONEY_FORMAT] * 3, "the gap in the money column came back unformatted"
        assert formats["code"] == [TEXT_FORMAT] * 3, "the blanked-out cell came back unformatted"
        assert _raw_values(target, "Figures", "amount") == [A_LONG_FIGURE, None, 1234.56], (
            "a missing figure was written as something other than nothing"
        )
        assert _raw_values(target, "Figures", "code") == ["A", None, "C"], "an empty string reached the cell as text"

    def test_a_rule_about_a_type_reaches_that_type_alone_and_not_the_one_nearest_it(
        self,
        tmp_path: Path,
    ) -> None:
        """Three types ruled on at once, and a fourth dtype nobody can name sitting between them.

        Every type rule measured so far has been ``Float64``, which leaves the step from a declared
        name to a dtype tested in one place out of seven. ``Int64`` and ``String`` are asked here so
        that a rule reaching the columns of *its own* type is pinned rather than assumed, and so a
        rule about counts cannot be quietly answering for the money beside it.

        The Float32 column is the point of the test. It is writable, it is a decimal, and it is not
        ``Float64``: no declared name means it, so no rule can speak about it and the money rule must
        not spill onto it. A writer matching on 'is this floating point' rather than on the one
        declared dtype would format it, and the tab would look right -- while a column nobody ruled
        on had been rounded to two decimals in the reader's view of it.
        """
        target = tmp_path / "pack.xlsx"
        frame = pl.DataFrame(
            {
                "key": ["r1", "r2"],
                "amount": [A_LONG_FIGURE, -5.5],
                "ratio": pl.Series("ratio", [1.5, 2.5], dtype=pl.Float32),
                "count": pl.Series("count", [1, 2], dtype=pl.Int64),
            }
        )

        _write(
            {"one": frame},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [
                            {"type": "Float64", "format": MONEY_FORMAT},
                            {"type": "Int64", "format": WHOLE_NUMBER_FORMAT},
                            {"type": "String", "format": TEXT_FORMAT},
                        ],
                    }
                ],
            },
        )

        formats = _formats(target, "Figures")
        assert formats["amount"] == [MONEY_FORMAT] * 2, "the rule about decimals did not reach the decimal column"
        assert formats["count"] == [WHOLE_NUMBER_FORMAT] * 2, "the rule about whole numbers reached no whole number"
        assert formats["key"] == [TEXT_FORMAT] * 2, "the rule about text reached no text"
        assert formats["ratio"] == ["General"] * 2, "a rule about Float64 reached a column that is not one"

    def test_a_tab_stating_an_empty_list_of_rules_is_written_like_one_stating_none(
        self,
        tmp_path: Path,
    ) -> None:
        """No rules is a thing an author is allowed to write down, not only a thing they may omit.

        A generated pipeline file, or one edited down to its last rule and then past it, arrives
        carrying ``formats: []``. Declaring the field non-empty would refuse that file outright while
        every other test in this suite still passed, because no other test writes an empty list. The
        two spellings mean the same thing and are asserted here to produce the same pack.
        """
        target = tmp_path / "pack.xlsx"
        stated_empty = tmp_path / "stated_empty.xlsx"

        _write({"one": _formatted_rows()}, {"path": str(target), "sheets": [{"port": "one", "sheet": "Figures"}]})
        _write(
            {"one": _formatted_rows()},
            {"path": str(stated_empty), "sheets": [{"port": "one", "sheet": "Figures", "formats": []}]},
        )

        assert _formats(stated_empty, "Figures") == _formats(target, "Figures"), (
            "an empty list of rules was read as something other than no rules"
        )
        assert _formats(stated_empty, "Figures")["as_at"] == [DATE_FORMAT] * 3

    def test_a_rule_about_a_type_no_column_can_be_is_accepted_and_changes_nothing(
        self,
        tmp_path: Path,
    ) -> None:
        """``Time`` is a declared name and no cell can hold one, so the rule is legal and inert.

        Refusing it would tie this field to the list of dtypes a cell cannot hold, and the two would
        then have to be kept in step forever for no gain: a Time column is refused outright, long
        before its format could matter, so the rule can never be the thing that goes wrong. What the
        run must not do is stop -- and the pack has to come out exactly as it would have without the
        rule, with the rule beside it still working.
        """
        target = tmp_path / "pack.xlsx"

        _write(
            {"one": _formatted_rows()},
            {
                "path": str(target),
                "sheets": [
                    {
                        "port": "one",
                        "sheet": "Figures",
                        "formats": [
                            {"type": "Time", "format": "hh:mm:ss"},
                            {"type": "Float64", "format": MONEY_FORMAT},
                        ],
                    }
                ],
            },
        )

        formats = _formats(target, "Figures")
        assert formats["amount"] == [MONEY_FORMAT] * 3, "the rule beside the inert one stopped working"
        assert formats["as_at"] == [DATE_FORMAT] * 3, "a rule about times reached a date"
        assert formats["key"] == ["General"] * 3, "a rule matching no column reached one anyway"
        assert formats["count"] == ["General"] * 3


class TestRefusalsAboutHowNumbersAreShown:
    """A rule that cannot mean one thing is refused, and refused with the wrong part named.

    Three of the four are questions about the configuration and are answered without looking at a
    single row, so they are answered identically whether three rows arrived or none. The fourth --
    a rule naming a column the frame does not carry -- is a question about the schema, and a frame
    with no rows has a schema like any other: an empty period must not be the month a silent typo
    ships.
    """

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    @pytest.mark.parametrize(
        ("rule", "named"),
        [
            pytest.param(
                {"type": "Float64", "column": "amount", "format": MONEY_FORMAT},
                ("type", "column"),
                id="both_a_type_and_a_column",
            ),
            pytest.param({"format": MONEY_FORMAT}, ("type", "column"), id="neither_a_type_nor_a_column"),
            pytest.param(
                {"type": "Money", "format": MONEY_FORMAT},
                ("Money", *DECLARED_TYPES),
                id="a_type_that_is_not_one_of_the_declared_names",
            ),
        ],
    )
    def test_a_rule_that_cannot_mean_one_thing_is_refused_before_anything_is_written(
        self,
        tmp_path: Path,
        rule: dict[str, Any],
        named: tuple[str, ...],
        height: int,
    ) -> None:
        """Each of the three refusals names the part the author has to change.

        Exactly one of ``type`` and ``column``, never both and never neither, is the same cross-field
        shape ``join`` uses for ``keys`` against ``left_keys``. Both is two rules written as one, and
        which of them was meant is not this tool's guess to make. Neither is a format with nothing to
        apply it to. A mistyped type name is refused with the whole vocabulary listed, because the
        author's next move is to pick one of them and a refusal that does not list them sends a
        reader to the source.
        """
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _formatted_rows(height)},
                {
                    "path": str(target),
                    "sheets": [{"port": "one", "sheet": "Figures", "formats": [rule]}],
                },
            )

        message = str(caught.value)
        for word in named:
            assert word in message, f"the refusal does not name {word!r}: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize("height", [3, 0], ids=["rows_arrived", "no_rows_arrived"])
    def test_a_column_rule_naming_a_column_the_frame_does_not_carry_is_refused_by_name(
        self,
        tmp_path: Path,
        height: int,
    ) -> None:
        """Writing the rule asserts the column is there, so a rule about a column that is not is a defect.

        The alternative -- passing over a rule that matched nothing -- delivers an unformatted pack
        and says nothing, and the author reads the configuration, sees the rule, and looks anywhere
        but there. Asked of a zero-row frame too: a schema is a schema whether or not rows arrived,
        and an empty period must not be when a renamed column slips through unnoticed.
        """
        target = tmp_path / "pack.xlsx"

        with pytest.raises((ConfigError, DataError)) as caught:
            _write(
                {"one": _formatted_rows(height)},
                {
                    "path": str(target),
                    "sheets": [
                        {
                            "port": "one",
                            "sheet": "Figures",
                            "formats": [{"column": "Variance pct", "format": THREE_DECIMALS}],
                        }
                    ],
                },
            )

        message = str(caught.value)
        assert "Variance pct" in message, f"the refusal does not name the column that is not there: {message}"
        assert not target.exists(), "a refused write left a pack behind"

    @pytest.mark.parametrize(
        ("rules", "repeated"),
        [
            pytest.param(
                [
                    {"type": "Float64", "format": MONEY_FORMAT},
                    {"type": "Float64", "format": THREE_DECIMALS},
                ],
                "Float64",
                id="the_same_type_ruled_on_twice",
            ),
            pytest.param(
                [
                    {"column": "amount", "format": MONEY_FORMAT},
                    {"column": "amount", "format": THREE_DECIMALS},
                ],
                "amount",
                id="the_same_column_ruled_on_twice",
            ),
        ],
    )
    def test_a_type_or_a_column_ruled_on_twice_is_refused_rather_than_resolved(
        self,
        tmp_path: Path,
        rules: list[dict[str, Any]],
        repeated: str,
    ) -> None:
        """Two answers to one question, and picking either of them is a reading of the list.

        Resolving this quietly means choosing the first rule or the last, and whichever is chosen is
        a rule about ordering that the author has not been told and may not share. Left unrefused,
        the money column of a delivered pack ships shown to three decimals with nothing said about
        it anywhere -- the rule the author believed they had written is in the file, and so is the
        one that actually won.

        A repeat also says something worth saying back. Either the author thought the list was
        ordered, or they forgot the rule they had already written; both are worth one sentence, and
        the sentence has to carry the repeated word and both codes, because that is what turns 'two
        rules clash' into 'delete this line'.

        Both shapes of rule are asked. They resolve through separate lookups, so refusing the
        repeated type and silently resolving the repeated column is a live outcome, not a hypothesis.
        """
        target = tmp_path / "pack.xlsx"

        with pytest.raises(ConfigError) as caught:
            _write(
                {"one": _formatted_rows()},
                {
                    "path": str(target),
                    "sheets": [{"port": "one", "sheet": "Figures", "formats": rules}],
                },
            )

        message = str(caught.value)
        assert repeated in message, f"the refusal does not name what was ruled on twice: {message}"
        for code in (MONEY_FORMAT, THREE_DECIMALS):
            assert code in message, f"the refusal does not show the author both codes ({code!r}): {message}"
        assert not target.exists(), "a refused write left a pack behind"


class TestHowNumbersAreShownIsDeclaredAndDocumented:
    """The field as a definition file meets it: optional, spelled safely, and demonstrated."""

    def test_the_template_declares_formats_as_an_optional_list_of_one_kind_of_rule(self) -> None:
        """A shape the loader can type-check, and one vocabulary for the type names rather than two.

        The names come from the constant the cast step and the melt step already publish. A second
        list spelled out here would be identical on the day it was written and drifted by the second
        time either was touched.

        ``format`` is required and non-empty, because a rule that shows a column no particular way is
        a rule with nothing in it. ``type`` and ``column`` are both optional in the template and
        exactly one of them is required by the cross-field rule, which is the only place a template
        cannot say it.
        """
        template = WriteExcel().config_template()
        entry = template["sheets"].items

        assert entry is not None and entry.fields is not None
        assert set(entry.fields) == {"port", "sheet", "formats", "widths", "fit_widths", "wrap", "table", "header"}

        formats = entry.fields["formats"]
        assert formats.type == "array"
        assert formats.required is False, "a tab that states nothing is written the way it always was"
        assert formats.default is None, "an unstated list is unstated, not an empty one"

        rule = formats.items
        assert rule is not None and rule.type == "object", "a rule has a declared shape"
        assert rule.fields is not None
        assert set(rule.fields) == {"type", "column", "format"}
        assert rule.fields["type"].required is False, "exactly one of type and column is a cross-field rule"
        assert rule.fields["column"].required is False
        assert rule.fields["format"].required is True and rule.fields["format"].non_empty is True
        assert tuple(rule.fields["type"].enum or ()) == tuple(DECLARED_TYPES), "one vocabulary, one spelling"

        assert not set(rule.fields) & YAML_REWRITES, "a field a definition file rewrites is not a field"
        assert not {name.lower() for name in DECLARED_TYPES} & YAML_REWRITES
        assert expression_fields(template) == (), "a format code is not a column reference"

    def test_a_worked_example_shows_a_money_format_and_the_notes_say_the_code_is_passed_through(self) -> None:
        """The case the field was added for is the case the page demonstrates, and it really runs.

        Every example in this catalog is executed by the conformance check, so an example carrying
        ``formats`` is an example proving the field works as printed.

        The notes have to say the other half: this library does not parse a format code. It is in no
        position to know what a future spreadsheet accepts, and a code it rejected wrongly would
        block a pack for no reason, where a wrong code shows visibly in the file and corrupts no
        figure.
        """
        tool = WriteExcel()
        help_ = tool.help()

        rules = [
            rule
            for example in help_.examples
            for entry in example.configuration["sheets"]
            for rule in entry.get("formats", ())
        ]
        assert any(rule["format"] == "#,##0.00" for rule in rules), (
            "the case the field exists for is not the case the page shows"
        )

        documented = " ".join((help_.detailed_description, *help_.notes)).lower()
        assert "format" in documented
        assert any(
            phrase in documented for phrase in ("as written", "passed through", "does not check", "does not parse")
        ), "the notes do not say the format code is passed through unchecked"

        check_tool(tool)


# ---------------------------------------------------------------------------


def _cell_kinds(path: Path, sheet: str) -> dict[str, tuple[str, Any]]:
    """Each written cell of one tab, as the kind the file stores it as and what it holds.

    The kind is ``"f"`` for a formula and ``"s"`` for text. A formula cell is a ``<f>`` element in the
    file, so the spreadsheet program works it out when the pack is opened; a text cell is shown as
    written.
    """
    book = openpyxl.load_workbook(path)
    try:
        return {
            cell.coordinate: (cell.data_type, cell.value)
            for row in book[sheet].iter_rows()
            for cell in row
            if cell.value is not None
        }
    finally:
        book.close()


_STARTS_WITH_EQUALS = ("=1+1", '=HYPERLINK("http://example.invalid","open")')
_STARTS_WITH_OTHER_FORMULA_SIGNS = ("+1+1", "-1+1", "@SUM(A1)")


class TestTextThatLooksLikeAFormulaIsWrittenAsText:
    """A text value is written as text, even when it starts with a sign a spreadsheet reads as a formula.

    A spreadsheet program treats text typed in starting with ``=``, ``+``, ``-`` or ``@`` as a
    formula. A frame's text column can hold exactly that: a free-text note, a reference typed by a
    person, a value copied from another system. Written as a formula, the cell shows a computed
    result instead of the text that was in the frame. That is a wrong value in a delivered pack. It
    is also a formula-injection risk: a pack is sent to people, and a formula such as ``HYPERLINK``
    or an external lookup runs on their machine when they open it.

    Each case is checked with no template and with one, because the two paths open the workbook
    differently and a fix to one could miss the other.
    """

    @pytest.fixture(params=["no template", "template"])
    def target_and_template(self, request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, Path | None]:
        template = _template(tmp_path / "template.xlsx") if request.param == "template" else None
        return tmp_path / "pack.xlsx", template

    def _write_text(self, target: Path, template: Path | None, frame: pl.DataFrame) -> dict[str, tuple[str, Any]]:
        config: dict[str, Any] = {"path": str(target), "sheets": [{"port": "rows", "sheet": "Rows"}]}
        if template is not None:
            config["template"] = str(template)
        _write({"rows": frame}, config)
        return _cell_kinds(target, "Rows")

    @pytest.mark.parametrize("text", _STARTS_WITH_EQUALS)
    def test_a_text_value_starting_with_equals_is_written_as_text_not_a_formula(
        self, target_and_template: tuple[Path, Path | None], text: str
    ) -> None:
        """A text value starting with = is written as the text in the frame, never as a live formula."""
        target, template = target_and_template
        cells = self._write_text(target, template, pl.DataFrame({"note": [text]}))

        assert cells["A2"] == ("s", text), (
            f"the text {text!r} was written as {cells['A2']!r}; a kind of 'f' is a live formula the "
            f"reader's spreadsheet runs when the pack is opened"
        )

    @pytest.mark.parametrize("text", _STARTS_WITH_OTHER_FORMULA_SIGNS)
    def test_a_text_value_starting_with_plus_minus_or_at_is_written_as_text(
        self, target_and_template: tuple[Path, Path | None], text: str
    ) -> None:
        """Pinned: these are written as text today, and must stay that way."""
        target, template = target_and_template
        cells = self._write_text(target, template, pl.DataFrame({"note": [text]}))

        assert cells["A2"] == ("s", text)

    @pytest.mark.parametrize("name", _STARTS_WITH_OTHER_FORMULA_SIGNS)
    def test_a_column_name_starting_with_plus_minus_or_at_is_written_as_text(
        self, target_and_template: tuple[Path, Path | None], name: str
    ) -> None:
        """Pinned: a heading starting with one of these signs is written as text today."""
        target, template = target_and_template
        cells = self._write_text(target, template, pl.DataFrame({name: ["x"]}))

        assert cells["A1"] == ("s", name)

    @pytest.mark.parametrize("name", _STARTS_WITH_EQUALS)
    def test_a_column_name_starting_with_equals_is_written_as_text_not_a_formula(
        self, target_and_template: tuple[Path, Path | None], name: str
    ) -> None:
        """A column name starting with '=' is written to the heading cell as text, not a formula.

        A table takes its column names from the heading cells, so a heading must hold the name as
        written, with and without a template.
        """
        target, template = target_and_template
        cells = self._write_text(target, template, pl.DataFrame({name: ["x"]}))

        assert cells["A1"] == ("s", name)
