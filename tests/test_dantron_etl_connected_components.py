"""Tests for the ConnectedComponents tool: put every record linked by a chain of edges into one group.

The tool reads one input port, ``edges``. Each row names two endpoints, in the columns that config
fields ``left`` and ``right`` name. It writes one output port, ``rows``: one row per distinct node,
carrying the node and the id of the component it belongs to.

A fixed number of joins cannot do this. Two joins follow a chain two links deep; the third link is
lost. A component is the whole chain, however long it runs.

The rulings pinned here, each with its mirror case:

**The component id is the smallest node in the component.** Not the first node read, not a row
number. That is what makes the answer the same whatever order the edges arrive in, and the same on
every run. Pinned with integers where the smallest by number is not the smallest as text (9 and 10),
so an implementation that compares the text form fails.

**Order does not matter.** Shuffling the edges, or swapping which side of an edge a node is on,
gives an identical frame. This is the property the tool exists for, so it is checked against an
independent grouping over many shuffles, not one.

**A null links nothing.** Like ``join``, which follows SQL's ``=``, a null endpoint pairs with no
other null. ``(a, null)`` and ``(null, b)`` leave ``a`` and ``b`` in separate components. A row with
one null endpoint still records its other node, alone. A row with two contributes nothing, and no
null ever appears as a node.

**A self-edge is how an isolated node is written.** ``(a, a)`` puts ``a`` in a component of its
own. The mirror: a self-edge on a node that is linked elsewhere does not cut it off.

**Output is sorted by component, then node.** So a written file does not churn between runs.

Vocabulary is neutral: items linked to other items.
"""

import random
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.tool import (
    CATEGORIES,
    REGISTRY,
    ConfigError,
    Ports,
    check_registry,
    check_tool,
    expression_fields,
    run_tool,
)
from dantron_etl.tools.connected_components import ConnectedComponents

_CONFIG: dict[str, Any] = {"left": "item", "right": "linked_item"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _edges(pairs: Iterable[tuple[Any, Any]], dtype: type[pl.DataType] | pl.DataType = pl.Int64) -> pl.DataFrame:
    """An edges frame with the two endpoint columns the default test config names."""
    pairs = list(pairs)
    return pl.DataFrame(
        {
            "item": [left for left, _ in pairs],
            "linked_item": [right for _, right in pairs],
        },
        schema={"item": dtype, "linked_item": dtype},
    )


def _run(edges: pl.DataFrame, config: Mapping[str, Any] | None = None) -> pl.DataFrame:
    """Run the tool through the boundary, which is the only way a tool is ever run."""
    result = run_tool(ConnectedComponents(), Ports({"edges": edges}), dict(config or _CONFIG))
    assert result.names == ("rows",)
    return result["rows"]


def _expected(
    pairs: Mapping[Hashable, Hashable] | Iterable[tuple[Any, Any]],
    dtype: type[pl.DataType] | pl.DataType = pl.Int64,
    node: str = "node",
    component: str = "component",
) -> pl.DataFrame:
    """The frame a reader expects, from (node, component) pairs, sorted by (component, node)."""
    rows = sorted(dict(pairs).items(), key=lambda item: (item[1], item[0]))
    return pl.DataFrame(
        {node: [n for n, _ in rows], component: [c for _, c in rows]},
        schema={node: dtype, component: dtype},
    )


def _oracle(pairs: Iterable[tuple[Any, Any]]) -> dict[Any, Any]:
    """An independent grouping: a plain union-find, written for clarity rather than speed.

    A null endpoint links nothing; its other endpoint, if any, is still a node.
    """
    parent: dict[Any, Any] = {}

    def find(node: Any) -> Any:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def add(node: Any) -> None:
        parent.setdefault(node, node)

    for left, right in pairs:
        if left is not None:
            add(left)
        if right is not None:
            add(right)
        if left is not None and right is not None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

    return {node: find(node) for node in parent}


def _random_pairs(seed: int) -> list[tuple[int | None, int | None]]:
    """A messy graph: several components, cycles, duplicates, self-edges and null endpoints."""
    rng = random.Random(seed)
    nodes = rng.sample(range(1, 500), 80)
    pairs: list[tuple[int | None, int | None]] = []
    for _ in range(70):
        pairs.append((rng.choice(nodes), rng.choice(nodes)))
    pairs.extend((node, node) for node in rng.sample(nodes, 10))
    pairs.extend(rng.sample(pairs, 15))  # duplicates
    pairs.extend([(rng.choice(nodes), None), (None, rng.choice(nodes)), (None, None)])
    return pairs


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestConnectedComponentsGrouping:
    """Which nodes end up together, and what the group is called."""

    def test_a_chain_puts_every_node_in_one_component_named_by_its_smallest_member(self) -> None:
        """Chain 12 - 10 - 30 - 9 - 25: no edge links 12 to 9 directly, and 9 is mid-chain.

        The first edge's left node is 12, so a tool naming the component after the first node it
        reads gets 12. As text, "10" sorts before "9", so a tool comparing the text form gets 10.
        The only right answer is 9.
        """
        edges = _edges([(12, 10), (30, 10), (30, 9), (25, 9)])

        result = _run(edges)

        assert_frame_equal(result, _expected({9: 9, 10: 9, 12: 9, 25: 9, 30: 9}))

    def test_nodes_with_no_chain_between_them_stay_in_separate_components(self) -> None:
        """Two chains, and an edge on each side of them: linking must not leak between groups."""
        edges = _edges([(1, 2), (2, 3), (7, 8), (8, 4)])

        result = _run(edges)

        assert_frame_equal(result, _expected({1: 1, 2: 1, 3: 1, 4: 4, 7: 4, 8: 4}))
        assert result["component"].n_unique() == 2

    def test_a_cycle_and_repeated_edges_give_each_node_exactly_once(self) -> None:
        """A cycle must not loop forever, and an edge given three times is still one link."""
        edges = _edges([(5, 6), (6, 7), (7, 5), (6, 5), (5, 6), (5, 6), (7, 7)])

        result = _run(edges)

        assert_frame_equal(result, _expected({5: 5, 6: 5, 7: 5}))
        assert result["node"].is_unique().all(), "a node appears once, however many edges name it"

    def test_a_self_edge_isolates_a_node_but_does_not_cut_off_a_linked_one(self) -> None:
        """(4, 4) is node 4 alone. (2, 2) beside (1, 2) leaves 2 in the component with 1."""
        edges = _edges([(4, 4), (2, 2), (1, 2)])

        result = _run(edges)

        assert_frame_equal(result, _expected({1: 1, 2: 1, 4: 4}))

    def test_a_null_endpoint_keeps_the_other_node_alone_and_links_nothing(self) -> None:
        """(3, null) and (null, 8) must not join 3 and 8 through the null: a null pairs with nothing.

        Each keeps its non-null node, alone. (null, null) adds nothing. No null node is written.
        The real edge (8, 11) shows a node seen beside a null still joins its genuine neighbours.
        """
        edges = _edges([(3, None), (None, 8), (None, None), (8, 11)])

        result = _run(edges)

        assert_frame_equal(result, _expected({3: 3, 8: 8, 11: 8}))
        assert result["node"].null_count() == 0
        assert result["component"].null_count() == 0

    def test_edges_whose_endpoints_are_all_null_give_no_rows_but_keep_the_columns(self) -> None:
        """Both-null rows contribute nothing, and an empty result is still typed like a full one."""
        for edges in (_edges([(None, None), (None, None)]), _edges([])):
            result = _run(edges)

            assert result.height == 0
            assert result.schema == pl.Schema({"node": pl.Int64, "component": pl.Int64})


# ---------------------------------------------------------------------------
# Order independence and ordering
# ---------------------------------------------------------------------------


class TestConnectedComponentsOrder:
    """The same edges give the same frame, however they arrive."""

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_the_result_is_identical_whatever_order_the_edges_arrive_in(self, seed: int) -> None:
        """Shuffle the rows and swap the sides of half the edges: the frame must not change.

        Each result is also checked against an independent grouping, so a tool that is
        consistently wrong cannot pass by agreeing with itself.
        """
        pairs = _random_pairs(seed)
        expected = _expected(_oracle(pairs))
        rng = random.Random(seed * 1000)

        first = _run(_edges(pairs))
        assert_frame_equal(first, expected)

        for _ in range(10):
            shuffled = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
            rng.shuffle(shuffled)

            assert_frame_equal(_run(_edges(shuffled)), first)

    def test_output_is_sorted_by_component_then_node(self) -> None:
        """Edges arrive largest first; rows leave in (component, node) order."""
        edges = _edges([(90, 80), (50, 60), (70, 40), (20, 90), (10, 50)])

        result = _run(edges)

        assert result.rows() == [(10, 10), (50, 10), (60, 10), (20, 20), (80, 20), (90, 20), (40, 40), (70, 40)]

    def test_a_chain_of_twenty_thousand_nodes_is_followed_to_its_far_end(self) -> None:
        """One line of 20,000 nodes, edges shuffled, smallest node at one end.

        A recursive walk fails on depth here. A walk that gives up after a fixed number of steps
        names the far end's component after some middle node. Every node must carry 0.
        """
        length = 20_000
        pairs = [(i + 1, i) for i in range(length - 1)]
        random.Random(7).shuffle(pairs)

        result = _run(_edges(pairs))

        assert result.height == length
        assert result["node"].to_list() == list(range(length))
        assert (result["component"] == 0).all()


# ---------------------------------------------------------------------------
# Columns written
# ---------------------------------------------------------------------------


class TestConnectedComponentsColumns:
    """What the output carries: two columns, named by config, typed like the endpoints."""

    def test_text_nodes_take_configured_column_names_and_other_columns_are_not_carried(self) -> None:
        """The output is the node and its component, nothing else, in the endpoints' own type.

        ``evidence`` is on the input and must not reach the output: a row there is a node, not an
        edge. The smallest text node names the component.
        """
        edges = pl.DataFrame(
            {
                "doc": ["D-200", "D-300", "D-150", "D-900"],
                "other_doc": ["D-300", "D-150", "D-150", "D-901"],
                "evidence": ["reference", "allocation", "self", "manual"],
            }
        )
        config = {"left": "doc", "right": "other_doc", "node_column": "doc_id", "component_column": "group_id"}

        result = _run(edges, config)

        expected = _expected(
            {"D-150": "D-150", "D-200": "D-150", "D-300": "D-150", "D-900": "D-900", "D-901": "D-900"},
            dtype=pl.String,
            node="doc_id",
            component="group_id",
        )
        assert_frame_equal(result, expected)
        assert result.columns == ["doc_id", "group_id"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _refusal_cases() -> list[Any]:
    """(edges, config, the text the refusal must quote)."""
    ints = _edges([(1, 2), (2, 3), (4, 4)])
    mixed = pl.DataFrame({"item": [1, 2, 4], "linked_item": ["2", "3", "4"]})
    return [
        pytest.param(ints, {"left": "missing_col", "right": "linked_item"}, "missing_col", id="left_not_on_edges"),
        pytest.param(ints, {"left": "item", "right": "missing_col"}, "missing_col", id="right_not_on_edges"),
        pytest.param(mixed, _CONFIG, "linked_item", id="left_and_right_of_different_types"),
        pytest.param(
            ints, {**_CONFIG, "node_column": "same", "component_column": "same"}, "same", id="one_name_for_both_outputs"
        ),
        pytest.param(ints, {**_CONFIG, "node_column": "   "}, "node_column", id="blank_node_column"),
        pytest.param(ints, {**_CONFIG, "component_column": "   "}, "component_column", id="blank_component_column"),
        pytest.param(ints, {**_CONFIG, "component_column": ""}, "component_column", id="empty_component_column"),
        pytest.param(ints, {"right": "linked_item"}, "left", id="no_left_at_all"),
        pytest.param(ints, {"left": "item"}, "right", id="no_right_at_all"),
        pytest.param(ints, {**_CONFIG, "depth": 5}, "depth", id="a_field_nobody_declared"),
    ]


class TestConnectedComponentsRefusals:
    """Refused by name, off the schema, before any row is read."""

    @pytest.mark.parametrize(("edges", "config", "named"), _refusal_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["edges_with_rows", "edges_with_no_rows"])
    def test_a_configuration_the_edges_cannot_satisfy_is_refused_by_name(
        self,
        edges: pl.DataFrame,
        config: Mapping[str, Any],
        named: str,
        height: int,
    ) -> None:
        """Each refusal names what is wrong, and an empty frame is refused exactly like a full one.

        Different endpoint types are refused rather than coerced: the integer 2 and the text "2"
        are not the same record, and quietly linking or not linking them decides which group a
        record lands in.
        """
        with pytest.raises(ConfigError) as caught:
            _run(edges.head(height), config)

        message = str(caught.value)
        assert named in message, f"the refusal must name {named!r}: {message}"

    def test_the_same_shapes_with_the_mistake_removed_are_accepted(self) -> None:
        """The mirror of every refusal above: a rule that refuses too much is as broken."""
        ints = _edges([(1, 2), (2, 3), (4, 4)])
        texts = pl.DataFrame({"item": ["1", "2", "4"], "linked_item": ["2", "3", "4"]})

        for edges, config in (
            (ints, _CONFIG),
            (texts, _CONFIG),
            (ints, {**_CONFIG, "node_column": "a", "component_column": "b"}),
            (ints, {**_CONFIG, "node_column": "item", "component_column": "linked_item"}),
        ):
            for height in (3, 0):
                result = _run(edges.head(height), config)
                assert result.width == 2


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------


class TestConnectedComponentsInTheCatalog:
    """Ports, template and examples meet the contract every tool meets."""

    def test_the_tool_declares_its_ports_template_and_examples(self) -> None:
        """One port in, one port out, two required endpoint columns and two defaulted output names."""
        tool = ConnectedComponents()

        check_tool(tool)

        assert tool.name == "connected_components"
        assert tool.category == "transform"
        assert tool.category in CATEGORIES
        assert tuple(tool.input_ports) == ("edges",)
        assert tuple(tool.output_ports) == ("rows",)

        template = tool.config_template()
        assert set(template) == {"left", "right", "node_column", "component_column"}
        for name in ("left", "right"):
            assert template[name].type == "string", name
            assert template[name].required is True, f"{name}: an edge has no default endpoint"
            assert template[name].non_empty is True, name
        assert template["node_column"].type == "string"
        assert template["node_column"].default == "node"
        assert template["component_column"].type == "string"
        assert template["component_column"].default == "component"
        assert expression_fields(template) == (), "endpoints are columns, not conditions"

        help_ = tool.help()
        assert help_.examples, "a worked example is required"
        for example in help_.examples:
            assert set(example.inputs.names) == {"edges"}
            assert set(example.outputs.names) == {"rows"}
        assert any(
            example.outputs["rows"].group_by(example.outputs["rows"].columns[1]).len()["len"].max() >= 3
            for example in help_.examples
        ), "an example must show a chain of at least three nodes, or it shows a join, not a closure"
        notes = " ".join(help_.notes).lower()
        assert "null" in notes, "the null ruling must be stated in the notes"
        assert "smallest" in notes, "how a component is named must be stated in the notes"

    def test_the_tool_is_in_the_shipped_catalog_and_the_catalog_still_conforms(self) -> None:
        """Registered under its machine name, and the whole catalog stays conformant."""
        assert "connected_components" in REGISTRY
        assert isinstance(REGISTRY.get("connected_components"), ConnectedComponents)

        check_registry(REGISTRY)
