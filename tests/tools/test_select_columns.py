"""Tests for the SelectColumns tool: which columns survive a step, and in which order.

SelectColumns takes one input port, ``rows``, and produces one, ``rows``: the same rows, in the
same order, with the same dtypes, carrying a different set of columns. It is the narrowest tool in
the library and it earns its place because a flow that carries every column it ever read is a flow
whose every later step is read against a schema nobody can hold in their head.

Four decisions are pinned here beyond the happy path, because each has a silent failure mode:

**Order follows the list, not the frame.** A pattern list is also a reordering, so ``[amount, code]``
puts ``amount`` first even though the frame does not. Under ``action: drop`` there is no list to
follow for the survivors, so they keep the input's order -- pinned deliberately, because "the order
listed" says nothing about the columns that were not listed.

**A pattern that matches nothing is refused, wildcard included.** A literal miss was already
refused by the tool this one is derived from; a wildcard that matched nothing silently contributed
zero columns, which is a whitelist failing open -- the same class of defect as a template accepting
a field type nothing knows how to check. Both spellings of "the schema moved under this config" are
refused here, and the refusal quotes the pattern, because the pattern is what the author has to go
and fix.

**Selecting nothing is refused in the tool's own words.** The boundary already refuses an output
port with no columns, but that refusal talks about ports; an author who dropped every column, or
passed an empty list, needs to be told about ``columns``. The discriminator pinned below is the
error type: :class:`ConfigError` from the tool, not :class:`PortError` from the envelope.

**Matching is case-sensitive.** ``fnmatch.fnmatch`` is case-*insensitive* on some platforms and
``fnmatch.fnmatchcase`` is not, so this is one wrong import away from a tool that quietly keeps a
column the author did not ask for, on one operating system only.

Refusals are parametrised over a populated frame and a frame with no rows at all, which is what
proves a refusal is read off the schema rather than off the data. Vocabulary is deliberately
generic -- record_id, code, region, dates, amount -- because the library carries no caller's nouns.
"""

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.authoring import CATEGORIES, ConfigError, Ports, REGISTRY, check_tool, run_tool
from dantron_etl.config import expression_fields
from dantron_etl.tools.select_columns import SelectColumns

# Values YAML turns into booleans before a tool ever sees them; no enum in this library may use one.
_YAML_BOOLEANS: Final[frozenset[str]] = frozenset({"true", "false", "yes", "no", "on", "off"})


def _rows() -> pl.DataFrame:
    """One frame with the column-name shapes every wildcard form needs to be tested against.

    Two columns sharing a prefix (``code``, ``code_suffix``), two sharing a suffix (``start_date``,
    ``end_date``), one matched only from the middle (``record_id``), and two dtypes that are not
    strings, so a selection that quietly rebuilt the frame instead of slicing it would show up.
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


def _selection_cases() -> list[Any]:
    """(config, the columns the output must carry, in order) for each selection worth pinning."""
    return [
        pytest.param(
            {"columns": ["amount", "code"]},
            ["amount", "code"],
            id="literals_appear_in_the_order_listed_not_the_frames_order",
        ),
        pytest.param({"columns": ["code*"]}, ["code", "code_suffix"], id="a_prefix_wildcard"),
        pytest.param(
            {"columns": ["*_date", "record_id"]},
            ["start_date", "end_date", "record_id"],
            id="a_suffix_wildcard_then_a_literal",
        ),
        pytest.param({"columns": ["*id*"]}, ["record_id"], id="a_wildcard_matched_from_the_middle"),
        pytest.param(
            {"columns": ["code*", "code", "*_date", "start_date"]},
            ["code", "code_suffix", "start_date", "end_date"],
            id="overlapping_patterns_are_de_duplicated_at_first_appearance",
        ),
        pytest.param(
            {"columns": ["*"]},
            ["record_id", "code", "code_suffix", "region", "start_date", "end_date", "amount"],
            id="a_bare_wildcard_keeps_everything_in_the_frames_order",
        ),
        pytest.param(
            {"columns": ["amount", "code*"], "action": "drop"},
            ["record_id", "region", "start_date", "end_date"],
            id="drop_leaves_the_survivors_in_the_frames_order",
        ),
        pytest.param(
            {"columns": ["*_date"], "action": "drop"},
            ["record_id", "code", "code_suffix", "region", "amount"],
            id="drop_by_wildcard",
        ),
        pytest.param(
            {"columns": ["record_id", "amount"], "action": "keep"},
            ["record_id", "amount"],
            id="keep_is_the_default_and_may_also_be_said_out_loud",
        ),
    ]


class TestSelectColumnsSelection:
    """The column set and its order change; nothing about the rows does."""

    @pytest.mark.parametrize(("config", "expected"), _selection_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_only_the_column_set_and_its_order_change(
        self,
        config: Mapping[str, Any],
        expected: Sequence[str],
        height: int,
    ) -> None:
        """The output is exactly the input's own selection: same rows, same order, same dtypes.

        Comparing against ``frame.select(expected)`` pins five things in one assertion -- which
        columns, in which order, with which dtypes, holding which rows, in which row order -- and
        running every case a second time against a frame with no rows shows the selection is made
        from the schema rather than from the data.
        """
        frame = _rows().head(height)

        result = run_tool(SelectColumns(), Ports({"rows": frame}), config)

        assert result.names == ("rows",), "one input, one output, both named for what they carry"
        assert result["rows"].columns == list(expected)
        assert_frame_equal(result["rows"], frame.select(expected))
        assert result["rows"].height == frame.height, "a column selection never drops a row"

    def test_a_pattern_is_case_sensitive(self) -> None:
        """``code*`` must not reach ``CODE_SUFFIX``.

        ``fnmatch.fnmatch`` folds case on some platforms and ``fnmatch.fnmatchcase`` never does, so
        the wrong one of those two names gives a tool that keeps an extra column on one operating
        system and not another -- the least reproducible defect available.
        """
        frame = pl.DataFrame({"code": ["X1"], "CODE_SUFFIX": ["a"], "amount": [10.0]})

        lower = run_tool(SelectColumns(), Ports({"rows": frame}), {"columns": ["code*"]})
        upper = run_tool(SelectColumns(), Ports({"rows": frame}), {"columns": ["CODE*"]})

        assert lower["rows"].columns == ["code"]
        assert upper["rows"].columns == ["CODE_SUFFIX"]

    def test_select_columns_leaves_its_input_untouched(self) -> None:
        """The boundary compares input schemas; the frame itself must be unchanged too.

        A tool that narrowed its input in place would pass a boundary check that only compared the
        schema of a frame it had already replaced, and would corrupt the other branch of any fan-out.
        """
        frame = _rows()
        before = frame.clone()

        run_tool(SelectColumns(), Ports({"rows": frame}), {"columns": ["code*"], "action": "drop"})

        assert_frame_equal(frame, before)


def _refusal_cases() -> list[Any]:
    """(config, the text its refusal must quote) for each way a selection can be wrong."""
    return [
        pytest.param({"columns": ["amount", "quantity"]}, "quantity", id="a_literal_column_that_is_not_there"),
        pytest.param({"columns": ["Amount"]}, "Amount", id="a_literal_column_whose_case_is_wrong"),
        pytest.param({"columns": ["temp_*"]}, "temp_*", id="a_wildcard_matching_nothing_while_keeping"),
        pytest.param(
            {"columns": ["temp_*"], "action": "drop"},
            "temp_*",
            id="a_wildcard_matching_nothing_while_dropping",
        ),
        pytest.param({"columns": []}, "columns", id="an_empty_pattern_list_selects_nothing"),
        pytest.param({"columns": ["*"], "action": "drop"}, "columns", id="dropping_every_column_selects_nothing"),
        pytest.param({"columns": [7]}, "columns[0]", id="a_pattern_that_is_not_text"),
    ]


class TestSelectColumnsRefusals:
    """What SelectColumns refuses, and in whose words: its own, off the schema, before any row."""

    @pytest.mark.parametrize(("config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_a_selection_the_schema_cannot_satisfy_is_refused_by_name(
        self,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Every one of these is a config authored against a schema that has since moved.

        :class:`ConfigError` rather than :class:`PortError` is the load-bearing half: a selection
        that ends up empty would also be caught by the envelope's "no columns at all" rule, but that
        refusal talks about output ports, and the author needs to be told about ``columns``. The two
        error types are siblings, so asking for the narrower one here pins which layer spoke.
        """
        frame = _rows().head(height)

        with pytest.raises(ConfigError) as caught:
            run_tool(SelectColumns(), Ports({"rows": frame}), config)

        message = str(caught.value)
        assert named in message, f"refusal should name {named!r}: {message}"

    def test_the_template_declares_a_typed_pattern_list_and_a_yaml_safe_action(self) -> None:
        """``items`` is what makes the element type real, and no enum value may be a YAML boolean.

        Without a declared element type, ``columns`` is one untyped array, so a numeric pattern
        reaches the matcher and fails as a ``TypeError`` from the standard library rather than as a
        refusal naming the field. And an enum spelled ``keep``/``drop`` survives a YAML file, where
        an obvious pair like ``yes``/``no`` would arrive as booleans and be refused with a type
        error that has nothing to do with what the author meant.
        """
        template = SelectColumns().config_template()

        assert set(template) == {"columns", "action"}
        assert template["columns"].type == "array"
        assert template["columns"].required is True
        assert template["columns"].items is not None, "an array with no declared element type checks nothing"
        assert template["columns"].items.type == "string"
        assert template["columns"].non_empty is True, (
            "an empty pattern list is a value, so 'required' alone accepts it; the field has to say "
            "it needs content, or the refusal lives in this tool rather than at the boundary"
        )
        assert template["action"].default == "keep", "the common case needs no configuration"
        assert set(template["action"].enum or ()) == {"keep", "drop"}
        assert not (set(template["action"].enum or ()) & _YAML_BOOLEANS)
        assert expression_fields(template) == (), "a column pattern is not a condition; there is no seam here"


class TestSelectColumnsInTheCatalog:
    """Described, documented with examples that actually run, and resolvable by name."""

    def test_select_columns_conforms_and_its_examples_cover_both_actions(self) -> None:
        """Conformance runs every worked example, so a documented output is a checked output.

        The tool this one is derived from documented a wildcard example whose output nobody ever
        ran. Both shapes an author reaches for -- a wildcard, and "everything except these" -- have
        to be demonstrated by an example that executes.
        """
        tool = SelectColumns()

        check_tool(tool)

        assert tool.name == "select_columns"
        assert tool.category == "preparation", "changing the shape of a frame without touching a value"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("rows",)
        assert tuple(tool.output_ports) == ("rows",)

        examples = tool.help().examples
        assert len(examples) >= 2, "one plain selection, and one showing the other action"
        assert any("*" in "".join(example.configuration["columns"]) for example in examples), (
            "a wildcard is the feature most likely to be copied out of the catalog, so it must be run"
        )
        assert any(example.configuration.get("action") == "drop" for example in examples), (
            "drop is the half the derived-from tool could not do, so it must be demonstrated"
        )

    def test_select_columns_is_in_the_shipped_catalog(self) -> None:
        """Registered under its machine name. The whole catalog is checked once, in tests/core/test_registry.py."""
        assert "select_columns" in REGISTRY
        assert isinstance(REGISTRY.get("select_columns"), SelectColumns)
