"""Tests for the preview store: a finished run's ports, answered a window at a time.

The page draws a fixed number of rows per port because drawing them all is a page no browser lays
out. That makes every question of the form *does any row anywhere carry this* unanswerable from the
page, so the rows are written beside it and served a window at a time, with **filtering done over
the whole port** rather than over what was drawn.

That distinction is the thing these tests exist to protect. A filter that quietly ran over the
window would still return rows, still look right, and would answer a question about the page while
the reader believed it was answering one about the data -- which is the same class of defect as a
row cap that does not say it is a cap.

The counts are pinned for the same reason: ``total`` is what the port holds and ``matched`` is what
the filter left, and a page that showed one where it meant the other would mislead in the direction
of "there is less here than there is".

Vocabulary is deliberately generic. The library carries no caller's nouns.
"""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from dantron_etl.preview import MAX_WINDOW, PortStore, PreviewError, persist_ports


def _entries() -> pl.DataFrame:
    """Two hundred rows, so a window is a genuine slice rather than the whole thing.

    Deliberately shaped: `code` repeats every four rows so a filter matches a known count; `label`
    carries a null so a filter never matches it and a cast never trips over it; `opened_on` is a
    real date, so a reader filtering on the text they can see is pinned rather than assumed.
    """
    codes = ["A-100", "A-200", "B-100", "B-200"]
    return pl.DataFrame(
        {
            "record_id": list(range(200)),
            "code": [codes[i % 4] for i in range(200)],
            "label": [None if i % 10 == 0 else f"row {i}" for i in range(200)],
            "amount": [round(i * 1.5, 2) for i in range(200)],
            "opened_on": [date(2026, 1 + (i % 12), 1 + (i % 28)) for i in range(200)],
        }
    )


def _panel(run: str, step: str) -> str:
    """One step's panel, as the page hands it over to be written down: one hidden section.

    Carries a non-ASCII character on purpose. A panel is full of a client's own names, and one that
    goes to disk and back through the wrong codec comes back with three characters where one was --
    which looks like bad data rather than like a file read the wrong way.
    """
    return (
        f'<section class="stepdata" data-step="{step}" hidden>'
        f'<div class="side" data-side="out"><span class="pill">{run} &middot; {step} &middot; caf\u00e9</span>'
        f'<table class="data" data-port="{step}.rows" data-lazy="1"><tbody></tbody></table></div>'
        f"</section>"
    )


@pytest.fixture
def store(tmp_path: Path) -> PortStore:
    """One persisted run holding two ports: a populated one and an empty one."""
    persist_ports(
        {"read.rows": _entries(), "read.rejected": _entries().head(0)},
        tmp_path / "ports",
    )
    return PortStore(tmp_path / "ports")


class TestWindowsOverAPort:
    """A window is a slice of the port, and says where in the port it came from."""

    def test_a_window_is_the_slice_asked_for_and_says_how_large_the_port_is(self, store: PortStore) -> None:
        found = store.window("read.rows", offset=0, limit=10)

        assert len(found.rows) == 10
        assert found.total == 200, "the port's own size, not the window's"
        assert found.matched == 200, "nothing was filtered, so everything matches"
        assert found.offset == 0
        assert found.columns == ("record_id", "code", "label", "amount", "opened_on")

    def test_successive_windows_walk_the_port_without_repeating_or_skipping(self, store: PortStore) -> None:
        """The property a pager rests on: page two begins exactly where page one stopped."""
        first = store.window("read.rows", offset=0, limit=50)
        second = store.window("read.rows", offset=50, limit=50)

        ids = [row[0] for row in first.rows] + [row[0] for row in second.rows]
        assert ids == [str(i) for i in range(100)]

    def test_a_window_past_the_end_is_empty_rather_than_a_refusal(self, store: PortStore) -> None:
        """Walking off the end is a reader pressing next once more, not a mistake worth stopping for."""
        found = store.window("read.rows", offset=500, limit=10)

        assert found.rows == ()
        assert found.total == 200

    def test_a_port_with_no_rows_still_reports_its_columns(self, store: PortStore) -> None:
        """An empty port is a real answer -- nothing was set aside -- and needs its shape to say so."""
        found = store.window("read.rejected")

        assert found.rows == ()
        assert found.total == 0
        assert found.columns == ("record_id", "code", "label", "amount", "opened_on")


class TestFilteringCoversTheWholePort:
    """The reason this store exists: a filter must see rows the page never drew."""

    def test_a_filter_matches_rows_beyond_the_first_window(self, store: PortStore) -> None:
        """50 of 200 rows carry the code, and only 10 were ever drawn -- the count must be 50."""
        found = store.window("read.rows", offset=0, limit=10, filters={"code": "A-100"})

        assert found.matched == 50, "every matching row in the port, not just the drawn ones"
        assert found.total == 200, "the port is still the size it was"
        assert len(found.rows) == 10, "the window is still a window"
        assert {row[1] for row in found.rows} == {"A-100"}

    def test_a_filter_applies_to_the_text_a_reader_can_see(self, store: PortStore) -> None:
        """A date filtered as the page prints it, not as the engine stores it."""
        found = store.window("read.rows", limit=MAX_WINDOW, filters={"opened_on": "2026-03-"})

        assert found.matched > 0
        assert all(row[4].startswith("2026-03-") for row in found.rows)

    def test_two_filters_narrow_together(self, store: PortStore) -> None:
        found = store.window("read.rows", limit=MAX_WINDOW, filters={"code": "A-", "label": "row 1"})

        assert found.matched > 0
        assert all(row[1].startswith("A-") and "row 1" in row[2] for row in found.rows)

    def test_a_filter_matching_nothing_says_so_without_pretending_the_port_is_empty(self, store: PortStore) -> None:
        found = store.window("read.rows", filters={"code": "no-such-code"})

        assert found.rows == ()
        assert found.matched == 0
        assert found.total == 200, "the port did not shrink because a search missed"

    def test_a_filter_is_a_substring_and_never_an_expression(self, store: PortStore) -> None:
        """Typed casually by somebody looking at a page, so it has no syntax to get wrong.

        Matched literally: the text a reader types is looked for in the value, and characters that
        mean something to a regular expression mean nothing here.
        """
        found = store.window("read.rows", filters={"code": "A-1.0"})

        assert found.matched == 0, "'.' is a character to look for, not a wildcard"

    def test_a_blank_filter_is_no_filter(self, store: PortStore) -> None:
        """An emptied search box must restore the port rather than matching every row by accident."""
        found = store.window("read.rows", filters={"code": "   "})

        assert found.matched == 200


class TestRefusals:
    """Each names what was asked for, because an empty answer reads exactly like a port that missed."""

    def test_an_unknown_port_is_refused_naming_the_ports_there_are(self, store: PortStore) -> None:
        with pytest.raises(PreviewError) as caught:
            store.window("read.nowhere")

        assert "read.nowhere" in str(caught.value)
        assert "read.rows" in str(caught.value)

    def test_a_filter_on_a_column_the_port_lacks_is_refused_naming_it(self, store: PortStore) -> None:
        """Silently ignoring it would answer with every row and look like a search that found them."""
        with pytest.raises(PreviewError) as caught:
            store.window("read.rows", filters={"quantity": "3"})

        assert "quantity" in str(caught.value)

    @pytest.mark.parametrize("limit", [0, -1, MAX_WINDOW + 1])
    def test_a_window_that_is_not_a_window_is_refused(self, store: PortStore, limit: int) -> None:
        with pytest.raises(PreviewError):
            store.window("read.rows", limit=limit)

    def test_a_negative_offset_is_refused(self, store: PortStore) -> None:
        with pytest.raises(PreviewError):
            store.window("read.rows", offset=-1)

    def test_a_directory_holding_no_run_is_refused_rather_than_read_as_empty(self, tmp_path: Path) -> None:
        """A mistyped directory must not look like a run that produced nothing."""
        with pytest.raises(PreviewError):
            PortStore(tmp_path / "nothing_here")


class TestRoutes:
    """One service, several flows, each at its own URL -- and what it says when asked for one it lacks."""

    @pytest.fixture
    def served(self, tmp_path: Path):
        """Two runs under one root, served on a port the operating system picks.

        Both carry panels, and both carry a step called ``read``: a panel is asked for by the step's
        own name, which is a name two flows are very likely to share.
        """
        from dantron_etl.preview import PAGE_NAME, persist_panels, serve

        root = tmp_path / "runs"
        for name, title in (("mrr", "Monthly recurring revenue"), ("loads", "Nightly load")):
            persist_ports({"read.rows": _entries()}, root / name)
            persist_panels({step: _panel(name, step) for step in ("read", "shape")}, root / name)
            (root / name / PAGE_NAME).write_text(f"<title>{title}</title>", encoding="utf-8")

        server = serve(root, host="127.0.0.1", port=0)
        yield f"http://127.0.0.1:{server.server_address[1]}"
        server.shutdown()
        server.server_close()

    def test_each_run_is_served_at_its_own_url(self, served: str) -> None:
        """The whole point: two flows for one client, read side by side, from one service."""
        import urllib.request

        for name, title in (("mrr", "Monthly recurring revenue"), ("loads", "Nightly load")):
            with urllib.request.urlopen(f"{served}/{name}/", timeout=5) as answer:
                assert answer.status == 200
                assert title.encode() in answer.read()

    def test_each_run_answers_for_its_own_ports_only(self, served: str) -> None:
        """A window is asked of a run, so two flows carrying a port of the same name cannot cross."""
        import json
        import urllib.request

        with urllib.request.urlopen(f"{served}/loads/api/rows?port=read.rows&limit=3", timeout=5) as answer:
            body = json.loads(answer.read())

        assert len(body["rows"]) == 3
        assert body["total"] == 200

    def test_a_run_asked_for_without_its_trailing_slash_is_redirected(self, served: str) -> None:
        """Without it a browser resolves the page's relative requests against the service root.

        Every window would then be asked of nobody, and the page would sit there looking static with
        no error to explain it -- so the redirect is what makes relative URLs safe to rely on.
        """
        import urllib.request

        with urllib.request.urlopen(f"{served}/mrr", timeout=5) as answer:
            assert answer.status == 200
            assert answer.url.endswith("/mrr/")

    def test_the_root_lists_every_run_that_has_been_written(self, served: str) -> None:
        """A reader who arrives without a flow in mind is shown what there is rather than a refusal."""
        import urllib.request

        with urllib.request.urlopen(served + "/", timeout=5) as answer:
            page = answer.read().decode()

        assert "Monthly recurring revenue" in page
        assert "Nightly load" in page
        assert 'href="/mrr/"' in page

    def test_an_unknown_run_says_which_runs_there_are(self, served: str) -> None:
        import json
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{served}/nowhere/", timeout=5)

        body = json.loads(caught.value.read())
        assert "nowhere" in body["error"]
        assert "mrr" in body["error"] and "loads" in body["error"]

    def test_a_steps_panel_is_handed_over_as_the_markup_it_was_written_as(self, served: str) -> None:
        """The reason the route exists: the page asks for one step's panel and puts it straight in.

        So it arrives as markup and not as JSON carrying markup -- there is nothing for a caller to
        decide about it -- and it arrives byte for byte, because the reader has to see exactly what
        they saw when all 672 panels were in the document.
        """
        import urllib.request

        with urllib.request.urlopen(f"{served}/mrr/api/panel?step=read", timeout=5) as answer:
            assert answer.status == 200
            assert "text/html" in answer.headers["Content-Type"]
            assert answer.read().decode("utf-8") == _panel("mrr", "read")

    def test_a_panel_is_answered_by_the_run_it_was_asked_of(self, served: str) -> None:
        """Two flows sharing a step name is ordinary; one answering for the other is not.

        This is the worst failure this route can have, and it is silent: a reader is shown one
        flow's rows under a step they chose in another, with the step's own name over the top of
        them, and nothing anywhere says the panel came from somewhere else.
        """
        import urllib.request

        for name in ("mrr", "loads"):
            with urllib.request.urlopen(f"{served}/{name}/api/panel?step=read", timeout=5) as answer:
                assert answer.read().decode("utf-8") == _panel(name, "read")

    def test_a_panel_asked_for_by_a_step_this_run_does_not_have_is_refused_by_name(self, served: str) -> None:
        """In the style of every other refusal here: what was asked for, and what is served instead.

        An empty body with a 200 would be indistinguishable from a step whose panel is empty, and
        the page would insert nothing and show an empty pane -- so a step the reader can select but
        whose panel cannot be had has to say so, once, in words a person can act on.
        """
        import json
        import urllib.error
        import urllib.request

        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(f"{served}/mrr/api/panel?step=nowhere", timeout=5)

        assert 400 <= caught.value.code < 500, "a step that does not exist is the caller's mistake, not a fault"
        body = json.loads(caught.value.read())
        assert "nowhere" in body["error"], "the refusal does not name the step that was asked for"
        assert "read" in body["error"] and "shape" in body["error"], (
            "the refusal does not say which steps this run does have, so a caller cannot tell a typo from a "
            "run that was written without panels"
        )

    def test_a_step_name_cannot_walk_out_of_the_run_it_was_asked_of(self, served: str) -> None:
        """A step id spells a filename, and every filename a caller writes is a path until it is checked.

        The sibling run is the test rather than /etc/passwd because it is reachable under whatever
        naming the panels are written with: if the step is pasted into a path, `../loads/read` lands
        on the other run's panel exactly. So this refuses the traversal and pins the run boundary at
        the same time -- and the service reads a run's directory per request, which is the one place
        a relative step id would be resolved.
        """
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(f"{served}/mrr/api/panel?step=../loads/read", timeout=5) as answer:
                served_back = answer.read().decode("utf-8")
        except urllib.error.HTTPError as refused:
            assert 400 <= refused.code < 500, "a step id that climbs out of the run is a refusal, not a fault"
            return
        assert _panel("loads", "read") not in served_back, (
            "a step id spelled as a relative path reached another run's panel; the same spelling reaches "
            "anything else on the disk the file naming happens to allow"
        )


class TestDiscovery:
    """Runs are found by looking, not by being listed somewhere a person has to remember to edit."""

    def test_a_directory_written_after_the_service_started_is_still_found(self, tmp_path: Path) -> None:
        """Discovery runs per request, so re-running a flow needs no restart."""
        from dantron_etl.preview import discover

        root = tmp_path / "runs"
        persist_ports({"read.rows": _entries()}, root / "first")
        assert [run.name for run in discover(root)] == ["first"]

        persist_ports({"read.rows": _entries()}, root / "second")
        assert [run.name for run in discover(root)] == ["first", "second"]

    def test_a_directory_holding_no_manifest_is_not_a_run(self, tmp_path: Path) -> None:
        """Something else living beside the runs must not be offered as one."""
        from dantron_etl.preview import discover

        root = tmp_path / "runs"
        persist_ports({"read.rows": _entries()}, root / "real")
        (root / "notes").mkdir()
        (root / "notes" / "scratch.txt").write_text("not a run", encoding="utf-8")

        assert [run.name for run in discover(root)] == ["real"]

    def test_a_run_takes_its_title_from_its_page(self, tmp_path: Path) -> None:
        """So the index reads as the flows a person named, not as directory names."""
        from dantron_etl.preview import PAGE_NAME, discover

        root = tmp_path / "runs"
        persist_ports({"read.rows": _entries()}, root / "mrr")
        (root / "mrr" / PAGE_NAME).write_text("<title>Monthly recurring revenue</title>", encoding="utf-8")

        assert discover(root)[0].title == "Monthly recurring revenue"

    def test_a_run_with_no_page_falls_back_to_its_directory_name(self, tmp_path: Path) -> None:
        """Ports written without a page are still reachable through the API rather than invisible."""
        from dantron_etl.preview import discover

        root = tmp_path / "runs"
        persist_ports({"read.rows": _entries()}, root / "headless")
        run = discover(root)[0]

        assert run.title == "headless"
        assert run.page is None


class TestPanelsWrittenBesideThePorts:
    """A step's panel is written down with the run's rows, and handed over when the step is opened.

    The page used to carry all of them: 672 panels came to 7.76 MB of an 8.98 MB document, with
    24,937 search boxes the browser laid out before it could paint, for the 671 steps nobody had
    clicked. So the panel follows the rows -- written once beside them, fetched when it is wanted --
    and this store is the half that has to hand back the one that was asked for and no other.
    """

    @pytest.fixture
    def written(self, tmp_path: Path) -> PortStore:
        """One run: its ports, and a panel for each of two steps."""
        from dantron_etl.preview import persist_panels

        persist_ports({"read.rows": _entries()}, tmp_path / "run")
        persist_panels({step: _panel("one", step) for step in ("read", "shape")}, tmp_path / "run")
        return PortStore(tmp_path / "run")

    def test_each_step_gets_back_the_panel_written_for_it_and_no_other(self, written: PortStore) -> None:
        """Byte for byte, because the markup is the whole contract: a panel is not re-rendered.

        And step by step, because the failure worth catching is not a panel that is missing -- that
        one is visible -- but a panel that is one step off, which puts one step's rows under
        another step's name and is believed.
        """
        for step in ("read", "shape"):
            assert written.panel(step) == _panel("one", step)
        assert written.panel("read") != written.panel("shape"), "the two steps come back as one panel"

    def test_a_step_with_no_panel_is_refused_naming_it_and_the_steps_there_are(self, written: PortStore) -> None:
        """Answering with nothing would draw an empty pane over a step that has data behind it."""
        with pytest.raises(PreviewError) as caught:
            written.panel("nowhere")

        assert "nowhere" in str(caught.value)
        assert "read" in str(caught.value) and "shape" in str(caught.value)

    def test_a_run_written_before_panels_existed_refuses_rather_than_failing(self, tmp_path: Path) -> None:
        """Every run already on disk is one of these, and the service reads whatever is written.

        A missing file is a `FileNotFoundError` on the way out of a request handler, which is a 500
        and a stack trace; what a reader needs to be told is that this run carries no panels, which
        is a fact about the run and not a fault in the service.
        """
        persist_ports({"read.rows": _entries()}, tmp_path / "old")

        with pytest.raises(PreviewError) as caught:
            PortStore(tmp_path / "old").panel("read")

        assert "read" in str(caught.value)


class TestAStepIdIsNotAFilename:
    """A step is named by whoever wrote the flow, and that name becomes a file and a query string.

    Both of those have syntax, and a step id is text: `join/lines` spells a directory, `%` spells
    an escape that has already happened, and `&` ends a query parameter. Nothing in the suite
    exercised either spelling -- every fixture step here is a bare word -- so the encoding on both
    sides could be dropped with all of it green, and the failure only turns up on a real flow where
    somebody named a tool after the thing it joins.
    """

    def test_a_step_id_that_spells_a_path_is_written_as_one_flat_file_and_read_back(self, tmp_path: Path) -> None:
        """Each panel is one file in one directory, whatever the step it answers to is called.

        Written straight, `entries/rows` lands in a subdirectory -- or, if that directory does not
        exist, nowhere at all -- and `a/b` and the already-escaped `a%2Fb` are two different steps
        that can be made to collide on one file, which serves one step's rows under the other's
        name. So the check is both halves: every step comes back as the markup written for it, and
        no two of them were ever written to the same place.
        """
        from dantron_etl.preview import PANEL_DIR, persist_panels

        steps = ["read", "entries/rows", "a/b", "a%2Fb", "../loads/read", "join lines & totals", "."]
        persist_ports({"read.rows": _entries()}, tmp_path / "run")
        index = persist_panels({step: _panel("one", step) for step in steps}, tmp_path / "run")
        store = PortStore(tmp_path / "run")

        for step in steps:
            assert store.panel(step) == _panel("one", step), f"{step!r} came back as another step's panel"
        assert len(set(index.values())) == len(steps), (
            f"{len(steps)} steps were written to {len(set(index.values()))} files, so two of them share one and "
            "the second overwrote the first"
        )
        written = list((tmp_path / "run" / PANEL_DIR).rglob("*"))
        assert len(written) == len(steps), f"the panels directory holds {len(written)} entries for {len(steps)} steps"
        assert all(len(item.relative_to(tmp_path / "run" / PANEL_DIR).parts) == 1 for item in written), (
            "a step id spelled a directory and the panel was written inside it; the service hands back what the "
            "index names, so anything it can be made to name is reachable"
        )

    def test_a_step_id_carrying_query_punctuation_is_answered_for_the_step_it_names(self, tmp_path: Path) -> None:
        """The page asks for a panel by putting the step into a query string, so the route has to take it back out.

        An `&` in a step id ends the parameter, a `#` ends the URL, and a `+` comes back as a space:
        each asks for a step that does not exist, or -- worse, and this is why it is pinned against
        two steps whose names share a prefix -- for a different one that does. The reader is shown
        another step's rows under the name of the step they clicked, and nothing says so.
        """
        import urllib.parse
        import urllib.request

        from dantron_etl.preview import persist_panels, serve

        steps = ["totals & rest", "totals", "a+b", "hash#mark"]
        root = tmp_path / "runs"
        persist_ports({"read.rows": _entries()}, root / "mrr")
        persist_panels({step: _panel("mrr", step) for step in steps}, root / "mrr")

        server = serve(root, host="127.0.0.1", port=0)
        try:
            served = f"http://127.0.0.1:{server.server_address[1]}/mrr/api/panel?step="
            for step in steps:
                asked = served + urllib.parse.quote(step, safe="")
                with urllib.request.urlopen(asked, timeout=5) as answer:
                    assert answer.read().decode("utf-8") == _panel("mrr", step), (
                        f"{step!r} was answered with another step's panel"
                    )
        finally:
            server.shutdown()
            server.server_close()
