"""Tests for the RenameColumns tool: a column changes its name and nothing else about it.

RenameColumns takes one input port, ``rows``, and produces one, ``rows``: the same rows, in the
same order, with the same dtypes, carrying the same columns in the same *positions* -- some of them
under different names. It is a sibling of SelectColumns, which already decides which columns
survive and in what order, and it exists because that tool cannot rename and no other tool can
either.

**Why the tool exists at all, and therefore what these tests are hardest about.** A rename can
almost be faked with two existing steps: a formula that computes a new column from the old one, and
a selection that drops the old one and lists the survivors in order. That is two steps instead of
one, and -- the part that matters -- the computed column is *appended*, so the faked rename moves
the column to the end of the frame. A consumer whose downstream comparator treats column order as
significant gets a different frame from the one it asked for, and gets it silently. So "the renamed
column stays exactly where it was" is not a nicety here, it is the entire justification for the
tool, and it is pinned twice below: once by slicing the columns either side of a rename in the
middle of a wide frame, and once by comparing the whole dtype sequence, which can only survive
unchanged if no column moved.

The rulings pinned beyond the happy path, each with a silent failure mode of its own:

**Renames are simultaneous, so a swap and a chain both work.** ``a -> b`` beside ``b -> a`` is
applied to the input schema as one mapping, not as two sequential steps, so neither rename can see
the other's output and no temporary name is needed. A naive implementation that renames one pair at
a time either raises a duplicate-column error from inside the query engine or, worse, ends with
both columns called the same thing.

**A collision is refused, and a name being renamed away is not a collision.** Writing over a column
that is staying is data loss with no warning, so it is refused by name. The same check must not
refuse the swap above, where the name being taken is being vacated in the very same step.

**A rename list that contradicts itself is refused before any frame is read.** Two entries claiming
the same new name, or two entries claiming the same old one, are an author's config half-edited;
either would otherwise resolve by whichever entry happened to be applied last.

**A rename that changes nothing is refused.** Renaming a column to the name it already has, a blank
name, and a whitespace-only name are all declarations that read as instructions and are not. That
is the same ruling the library already makes elsewhere -- a pattern matching nothing, a default that
is never applied, ``non_empty`` on a type with no notion of emptiness -- and it is made here for the
same reason: the usual cause is a config edited halfway.

**Every refusal is a ConfigError and every refusal names the offender.** There is no
:class:`~dantron_etl.authoring.DataError` anywhere in this file, deliberately: this tool never reads a
row, so there is no value it could ever diagnose, and every one of its refusals is decided off the
schema and the config alone. That is what the zero-row axis on both parametrised tests proves -- a
refusal that only fires on a populated frame was read off the data by accident.

Vocabulary is deliberately generic -- record_id, code, region, dates, amount -- because the library
carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.rename_columns import RenameColumns

# The config shape these tests fix: an ordered list of small objects, one per rename, the same shape
# `case_when` uses for its rules and `group_by` for its aggregations, and the shape a form renders
# from a declared element type. `from` is a Python keyword, which is deliberately irrelevant here:
# it is a mapping key and a YAML key, never an identifier, so nothing in the template, the
# validator, the defaults filler or a definition-file loader ever binds it as a name. It is also not
# a value, so the YAML-unsafe-value rule that governs enum members does not reach it either.
_FROM: Final[str] = "from"
_TO: Final[str] = "to"

# Values a YAML loader resolves into booleans before a tool ever sees them; no enum in this library
# may offer one. This tool is expected to declare no enum at all, which is the stronger position.
_YAML_BOOLEANS: Final[frozenset[str]] = frozenset({"true", "false", "yes", "no", "on", "off"})


def _rows() -> pl.DataFrame:
    """A frame wide enough that a column moving to the end is visible, with four distinct dtypes.

    Seven columns, so a rename in the middle has real columns on both sides of it; an integer, two
    kinds of text, dates and a float, so a rename that quietly rebuilt the frame -- casting, or
    reordering by dtype -- shows up in the dtype sequence rather than hiding behind uniform columns.
    """
    return pl.DataFrame(
        {
            "record_id": [11, 12, 13],
            "code": ["X1", "Y2", "Z3"],
            "code_suffix": ["a", "b", None],
            "region": ["north", "south", "north"],
            "start_date": [date(2024, 1, 1), date(2024, 2, 1), date(2024, 3, 1)],
            "end_date": [date(2024, 1, 31), date(2024, 2, 29), date(2024, 3, 31)],
            "amount": [10.0, -5.0, 0.0],
        }
    )


def _config(renames: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """The configuration that asks for ``renames``, as ``(old name, new name)`` pairs."""
    return {"columns": [{_FROM: old, _TO: new} for old, new in renames]}


def _expected(frame: pl.DataFrame, renames: Sequence[tuple[str, str]]) -> pl.DataFrame:
    """What ``frame`` must look like after ``renames``, built without using any rename of its own.

    Every column of the input is selected in the input's own order and aliased if it is being
    renamed, which is an independent statement of the property under test: had this been written as
    ``frame.rename(...)`` it would assert only that the tool calls the same library function the
    test does, and a tool that appended instead of renaming in place would still have to be caught
    by a separate assertion on column order.
    """
    lookup = dict(renames)
    return frame.select([pl.col(name).alias(lookup.get(name, name)) for name in frame.columns])


def _rename_cases() -> list[Any]:
    """``(old, new)`` pairs for each rename worth pinning on its own."""
    return [
        pytest.param(
            [("region", "territory")],
            id="a_column_in_the_middle_of_the_frame",
        ),
        pytest.param([("record_id", "identifier")], id="the_first_column"),
        pytest.param([("amount", "value")], id="the_last_column"),
        pytest.param(
            [("code", "label"), ("amount", "value"), ("start_date", "opened_on")],
            id="three_columns_renamed_in_one_step",
        ),
        pytest.param(
            [("amount", "value"), ("record_id", "identifier")],
            id="the_list_order_does_not_reorder_the_frame",
        ),
        pytest.param(
            [("code", "CODE")],
            id="a_rename_that_only_changes_case_is_a_real_rename",
        ),
        pytest.param([("start_date", "opened_on")], id="a_date_column_keeps_its_dtype"),
    ]


class TestRenameColumnsRenaming:
    """The names change. The positions, the dtypes, the values and the row order do not."""

    @pytest.mark.parametrize("renames", _rename_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_rename_changes_only_the_name(
        self,
        renames: Sequence[tuple[str, str]],
        height: int,
    ) -> None:
        """The output is the input with names swapped in place: same slots, dtypes, values, rows.

        The dtype-sequence assertion is the one that cannot be satisfied by a tool that appends a
        renamed column at the end: the sequence is read positionally, so any column that moved
        shows up there even when the set of names is right. Running every case again against a
        frame with no rows shows the rename is applied to the schema and not to the data -- a
        zero-row frame must come back zero-row with the new schema, not empty of columns and not
        refused.
        """
        frame = _rows().head(height)

        result = run_tool(RenameColumns(), Ports({"rows": frame}), _config(renames))

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert_frame_equal(result["rows"], _expected(frame, renames))
        assert result["rows"].dtypes == frame.dtypes, (
            "a rename moves no column, so the dtypes read positionally must be unchanged"
        )
        assert result["rows"].height == frame.height, "a rename never drops or adds a row"

    def test_a_rename_in_the_middle_of_a_wide_frame_leaves_every_other_column_where_it_was(self) -> None:
        """The headline property, sliced either side of the renamed column.

        This is the whole reason the tool exists rather than being faked with a formula and a
        selection: that fake appends the new column, so the renamed column arrives last and every
        column after the original's slot shifts down one. Both halves are asserted -- the new name
        occupies the old slot, and the columns on either side of it are untouched -- because either
        one alone is satisfied by a frame that is subtly wrong.
        """
        frame = _rows()
        slot = frame.columns.index("region")
        assert 0 < slot < frame.width - 1, "the fixture must put this column in the middle, or this proves nothing"

        result = run_tool(RenameColumns(), Ports({"rows": frame}), _config([("region", "territory")]))

        renamed = result["rows"]
        assert renamed.columns[slot] == "territory", "the new name sits in the old name's slot"
        assert renamed.columns[:slot] == frame.columns[:slot], "nothing ahead of the rename moved"
        assert renamed.columns[slot + 1 :] == frame.columns[slot + 1 :], "nothing behind the rename moved"
        assert renamed.columns[-1] != "territory", (
            "appending the renamed column is the exact defect this tool exists to avoid"
        )
        assert_frame_equal(renamed, _expected(frame, [("region", "territory")]))

    @pytest.mark.parametrize(
        "renames",
        [
            pytest.param([("code", "region"), ("region", "code")], id="a_swap_of_two_names"),
            pytest.param(
                [("code", "region"), ("region", "amount"), ("amount", "code")],
                id="a_three_way_cycle",
            ),
            pytest.param(
                [("code", "region"), ("region", "territory")],
                id="a_chain_where_one_rename_vacates_the_name_another_takes",
            ),
        ],
    )
    def test_names_are_exchanged_simultaneously(self, renames: Sequence[tuple[str, str]]) -> None:
        """RULING: a swap is supported, not refused, because the list is one mapping over one schema.

        Supported rather than refused because the meaning is unambiguous -- every entry reads the
        *input's* schema, none reads another entry's output -- and because refusing it would force
        an author to invent a temporary name and run the tool twice, which is exactly the two-step
        workaround this tool was built to delete. Refusing would also be arbitrary next to the chain
        case below, which is the same property under a different shape.

        The property is pinned rather than the mechanism: a pair-at-a-time implementation fails here
        either by raising a duplicate-column error out of the query engine or by producing two
        columns with the same name, and both are caught by comparing against a frame built by
        aliasing every column in one pass.
        """
        frame = _rows()

        result = run_tool(RenameColumns(), Ports({"rows": frame}), _config(renames))

        assert_frame_equal(result["rows"], _expected(frame, renames))
        assert result["rows"].dtypes == frame.dtypes, "an exchange of names moves no column either"

    def test_rename_columns_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame's values must be unchanged too.

        A tool that renamed its input in place would corrupt the other branch of any fan-out, and
        the boundary's own check compares the schema of whatever frame the port now holds -- so the
        input frame is compared here against a clone taken before the run.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(RenameColumns(), Ports({"rows": frame}), _config([("region", "territory")]))

        assert_frame_equal(frame, before)
        assert frame.columns == before.columns


def _refusal_cases() -> list[Any]:
    """``(config, the text its refusal must quote)`` for each way a rename list can be wrong."""
    return [
        pytest.param(
            _config([("quantity", "count")]),
            "quantity",
            id="an_old_name_the_frame_does_not_carry",
        ),
        pytest.param(
            _config([("Code", "label")]),
            "Code",
            id="an_old_name_whose_case_is_wrong",
        ),
        pytest.param(
            _config([("code", "region")]),
            "region",
            id="a_new_name_already_taken_by_a_column_that_is_staying",
        ),
        pytest.param(
            _config([("code", "label"), ("region", "label")]),
            "label",
            id="two_renames_claiming_the_same_new_name",
        ),
        pytest.param(
            _config([("code", "label"), ("code", "other_label")]),
            "code",
            id="two_renames_claiming_the_same_old_name",
        ),
        pytest.param(
            _config([("region", "region")]),
            "region",
            id="renaming_a_column_to_the_name_it_already_has",
        ),
        pytest.param(
            _config([("record_id", "identifier"), ("code", "")]),
            "columns[1]",
            id="a_blank_new_name",
        ),
        pytest.param(
            _config([("code", "   ")]),
            "columns[0]",
            id="a_whitespace_only_new_name",
        ),
        pytest.param(
            _config([("\t", "label")]),
            "columns[0]",
            id="a_whitespace_only_old_name",
        ),
        pytest.param({"columns": []}, "columns", id="an_empty_rename_list"),
        pytest.param(
            {"columns": [{_FROM: 7, _TO: "label"}]},
            "columns[0].from",
            id="an_old_name_that_is_not_text",
        ),
        pytest.param(
            {"columns": [{_FROM: "code"}]},
            "columns[0].to",
            id="an_entry_with_no_new_name_at_all",
        ),
    ]


class TestRenameColumnsRefusals:
    """What RenameColumns refuses, and in whose words: its own, off the schema, before any row."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_rename_that_cannot_be_honoured_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these is a config authored against a schema, or an author, that has moved on.

        :class:`ConfigError` throughout, and never :class:`~dantron_etl.authoring.DataError`: not one of
        these refusals looks at a value, so sending a reader to the data would send them to the
        wrong place. The load-bearing half of each case is the text quoted in the message. A
        collision reported as "a column could not be renamed" is a refusal somebody has to reproduce
        before they can fix it; the same refusal naming ``region`` is a refusal they fix by reading
        it. The two blank-name cases quote the entry's own path rather than a column name for
        exactly that reason -- there is no column name worth printing, so the entry's index is the
        only thing that locates the mistake, and a blank name reported as "column '   ' is not on
        this port" would send the reader looking for a column instead of a typo.

        Each case runs again against a frame with no rows, which is what shows the refusal came from
        the schema: a check that quietly reads the data would stop refusing the moment a frame
        arrived empty, and would then rename -- or fail to -- inside a flow nobody was watching.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(RenameColumns(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_the_template_declares_a_typed_list_of_name_pairs_that_must_not_be_empty(self) -> None:
        """``items`` and ``fields`` are what make the pair list real to everything outside the tool.

        Without a declared element type, ``columns`` is one untyped array: a numeric name reaches
        the rename and fails as an error from the query engine rather than as a refusal naming
        ``columns[0].from``, an entry missing its ``to`` is discovered inside ``execute``, and a form
        renderer has nothing to render but a free-text box.

        ``non_empty`` is asserted separately from ``required`` because they answer different
        questions and only one of them refuses ``[]``: an empty list is a value, so ``required``
        alone accepts it, and a tool whose template says only ``required`` would have to re-refuse
        the empty list in its own code -- or, more likely, would rename nothing and report success.

        ``required`` on the element spec itself must stay false: the catalog refuses it outright as
        a declaration nothing reads, since an element that is absent is not an element.
        """
        template = RenameColumns().config_template()

        assert set(template) == {"columns"}, "one field: the list of renames, and nothing else to configure"
        columns = template["columns"]
        assert columns.type == "array"
        assert columns.required is True
        assert columns.non_empty is True, (
            "an empty list is a value, so 'required' alone accepts it; the field has to say it needs content"
        )

        element = columns.items
        assert element is not None, "an array with no declared element type checks nothing inside itself"
        assert element.type == "object"
        assert element.required is False, "required on an element is refused by the catalog as unread"
        assert element.fields is not None, "an object with no declared members checks nothing inside itself"
        assert set(element.fields) == {_FROM, _TO}, "one entry names the column and the name it takes"
        assert all(member.type == "string" for member in element.fields.values())
        assert all(member.required is True for member in element.fields.values()), (
            "half an entry is not an entry; both halves are needed before the frame is even read"
        )
        assert all(member.enum is None for member in element.fields.values()), (
            "a column name is drawn from the frame, not from a published set"
        )
        assert not (set(element.fields) & _YAML_BOOLEANS), "no key here is a word YAML resolves to something else"
        assert expression_fields(template) == (), "a column name is not an expression; there is no seam here"


class TestRenameColumnsInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_rename_columns_conforms_and_its_examples_demonstrate_a_rename_in_place(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output.

        The examples are held to the property the tool exists for, not merely to running: an
        example whose documented output has a different width, or a different dtype sequence, from
        its own input is an example demonstrating something other than a rename -- and it is the
        page somebody copies. ``check_row_order`` is required on every example because this tool
        touches no row at all: leaving it off would let the one property a reader most wants to
        trust go unchecked in the catalog.
        """
        tool = RenameColumns()

        check_tool(tool)

        assert tool.name == "rename_columns"
        assert tool.category == "preparation", "changing the shape of a frame without touching a value"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert examples, "a catalog page with no example is a page nobody can copy"
        for example in examples:
            given = example.inputs["rows"]
            produced = example.outputs["rows"]
            assert produced.width == given.width, f"{example.scenario!r}: a rename adds and drops no column"
            assert produced.dtypes == given.dtypes, f"{example.scenario!r}: a rename moves no column"
            assert produced.columns != given.columns, f"{example.scenario!r}: nothing was renamed"
            assert example.check_row_order is True, (
                f"{example.scenario!r}: this tool preserves row order, so the example must have it checked"
            )

    def test_rename_columns_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "rename_columns" in REGISTRY
        assert isinstance(REGISTRY.get("rename_columns"), RenameColumns)
