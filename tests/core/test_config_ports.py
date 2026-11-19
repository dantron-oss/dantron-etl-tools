"""Tests for input ports a tool works out from its own configuration.

Today a tool's input ports are a static attribute. Every step that uses that tool has the same
ports, whatever it is configured to do. That is right for almost every tool in the catalog, and
wrong for one shape: a step that reads a different number of frames depending on how it is set up.
The one that forces the issue is a workbook writer. Eleven sheets should be one step reading eleven
named frames, not eleven chained steps each reopening the file.

So a tool may now answer the question itself. The member is optional. A tool that does not
implement it keeps the static attribute it always had, and behaves exactly as it did. That
sentence is the most important thing in this file, because twenty-six shipped tools depend on it.

What is pinned here, in order:

**Nothing moves for a tool without the member.** Checked against the real registry, not only
against a fixture, and checked at every place that reads a port: the boundary, the loader, the
runner, and the conformance check.

**A tool with the member derives its ports from the step's configuration.** Two steps using the
same tool with different configuration have different ports. A static attribute cannot say that,
which is the whole reason the member exists.

**Every reader of a port asks the same question.** The boundary checks the envelope against the
derived ports. The runner gathers exactly them and hands them to ``execute``. The loader validates
a step's edges against them, and its refusal lists the ports that configuration declares rather
than the static ones -- a message listing the static ports would send the author to look at the
wrong thing.

**The two traps.** A source step is one that reads nothing. That test used to be "the static
attribute is empty", which is now the wrong question twice over: a config-derived tool that reads
nothing for a given configuration is a source, and a config-derived tool that reads three frames is
not one, even though its static attribute is empty in both cases.

**The catalog.** A catalog page has no configuration in hand, so it cannot ask the step. The
decision pinned below is that a config-derived tool shows the ports of its first worked example,
and the catalog says that those ports came from a configuration. The argument is in
:class:`TestTheCatalogWithNoConfigurationInHand`.

The vocabulary here is deliberately generic -- parts, labels, seeds -- because this library carries
no caller's nouns.
"""

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl
import pytest

from dantron_etl.paths import project_root
from dantron_etl import PipelineError, PortRef, load_pipeline
from dantron_etl import RunError, run_pipeline
from dantron_etl.authoring import (
    ConfigField,
    ConformanceError,
    PortError,
    Ports,
    REGISTRY,
    Tool,
    ToolHelp,
    ToolRegistry,
    WorkedExample,
    check_tool,
    run_tool,
)
from dantron_etl.contract import catalog_input_ports, input_ports_for, ports_from_config

PROJECT_ROOT = project_root(__file__)

# ---------------------------------------------------------------------------
# Reference tools
#
# Three of them. One derives its ports from its configuration, and is the subject of most of this
# file. Two do not, and are here to hold the other half of the contract: that a tool written before
# any of this existed still behaves the way it always did.
# ---------------------------------------------------------------------------


def _seed_frame(value: str) -> pl.DataFrame:
    """Two rows carrying the seed's own name, so a test can tell which frame landed where."""
    return pl.DataFrame({"id": [1, 2], "seed": [value, value]})


class _SeedTool:
    """No inputs, one output. A source, by the old test and the new one alike."""

    name = "seed_rows"
    description = "Emit two rows carrying the configured value."
    category = "input"
    input_ports: Sequence[str] = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "value": ConfigField(
                type="string",
                description="The value both rows carry.",
                required=True,
                non_empty=True,
            )
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description="Emits a fixed pair of rows, so a test flow has somewhere to start.",
            examples=(
                WorkedExample(
                    scenario="Start a flow",
                    configuration={"value": "a"},
                    explanation="Two rows, both carrying 'a'.",
                    inputs=Ports(),
                    outputs=Ports({"rows": _seed_frame("a")}),
                    check_row_order=True,
                ),
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return Ports({"rows": _seed_frame(config["value"])})


class _PassTool:
    """One fixed input, one fixed output. The ordinary shape, and it must not change."""

    name = "pass_rows"
    description = "Hand the rows on with a column saying they passed through."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "mark": ConfigField(
                type="string",
                description="What to write in the mark column.",
                required=True,
                non_empty=True,
            )
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description="Adds one column holding the configured mark, and changes nothing else.",
            examples=(
                WorkedExample(
                    scenario="Mark every row",
                    configuration={"mark": "m"},
                    explanation="Every row gains a mark column holding 'm'.",
                    inputs=Ports({"rows": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"rows": pl.DataFrame({"id": [1], "mark": ["m"]})}),
                    check_row_order=True,
                ),
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return Ports({"rows": inputs["rows"].with_columns(pl.lit(config["mark"]).alias("mark"))})


class _SinkTool:
    """One fixed input, and a category that lets it end a flow. Static, like the rest of the shipped catalog."""

    name = "sink_rows"
    description = "Take the rows somewhere else, and hand the same rows on."
    category = "output"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description="Stands in for a sink, so a test flow has a legitimate last step.",
            examples=(
                WorkedExample(
                    scenario="End a flow",
                    configuration={},
                    explanation="The rows leave unchanged.",
                    inputs=Ports({"rows": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"rows": pl.DataFrame({"id": [1]})}),
                    check_row_order=True,
                ),
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return Ports({"rows": inputs["rows"]})


def _collect_examples() -> tuple[WorkedExample, ...]:
    """One worked example, reading two named parts. Its ports are the ones the catalog shows."""
    left = _seed_frame("a")
    right = _seed_frame("b")
    stacked = pl.concat(
        [
            left.with_columns(pl.lit("L").alias("part")),
            right.with_columns(pl.lit("R").alias("part")),
        ]
    )
    return (
        WorkedExample(
            scenario="Stack two named parts into one frame",
            configuration={
                "parts": [
                    {"port": "left", "label": "L"},
                    {"port": "right", "label": "R"},
                ]
            },
            explanation=(
                "The configuration names two parts, so this step reads two input ports. A step "
                "naming three parts would read three."
            ),
            inputs=Ports({"left": left, "right": right}),
            outputs=Ports({"rows": stacked}),
            check_row_order=True,
        ),
    )


class _CollectTool:
    """Reads one input port per part its configuration names, in the order they are written.

    The static attribute is empty, and deliberately so: this tool has no fixed ports at all. Every
    port it reads comes from a step's own configuration. That makes it the trap the runner has to
    survive -- an empty static attribute no longer means "reads nothing".

    It records what it was asked, and what it was handed, so a test can compare the two without
    reaching into the runner.
    """

    description = "Stack the frames on the named parts into one frame."
    category = "output"
    input_ports: Sequence[str] = ()
    output_ports = ("rows",)

    def __init__(
        self,
        *,
        name: str = "collect_parts",
        examples: Sequence[WorkedExample] | None = None,
    ) -> None:
        self.name = name
        self._examples = tuple(examples) if examples is not None else _collect_examples()
        #: Every port tuple this tool was asked to derive, in the order it was asked.
        self.derived: list[tuple[str, ...]] = []
        #: The port names actually handed to ``execute``, per call.
        self.executed: list[tuple[str, ...]] = []

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]:
        """The optional member: this step's input ports, read off this step's configuration."""
        ports = tuple(str(part["port"]) for part in config.get("parts", ()))
        self.derived.append(ports)
        return ports

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "parts": ConfigField(
                type="array",
                description="One entry per input port this step reads, in the order they are read.",
                required=True,
                items=ConfigField(
                    type="object",
                    description="One named part.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port this part arrives on.",
                            required=True,
                            non_empty=True,
                        ),
                        "label": ConfigField(
                            type="string",
                            description="What the part column says for this part's rows.",
                            required=True,
                            non_empty=True,
                        ),
                    },
                ),
            )
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every part named in the configuration arrives on its own input port. The frames "
                "are stacked in the order the parts are written, each carrying a column saying "
                "which part it came from. A configuration naming no parts reads nothing at all."
            ),
            examples=self._examples,
            notes=("The input ports of a step using this tool are the parts that step names.",),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self.executed.append(inputs.names)
        parts = list(config.get("parts", ()))
        if not parts:
            return Ports({"rows": pl.DataFrame(schema={"id": pl.Int64, "seed": pl.String, "part": pl.String})})
        stacked = pl.concat([inputs[part["port"]].with_columns(pl.lit(part["label"]).alias("part")) for part in parts])
        return Ports({"rows": stacked})


def _registry(collect: _CollectTool | None = None) -> ToolRegistry:
    """A registry holding only the reference tools, so no shipped tool is touched by these tests."""
    registry = ToolRegistry()
    registry.register(_SeedTool())
    registry.register(_PassTool())
    registry.register(_SinkTool())
    registry.register(collect if collect is not None else _CollectTool())
    return registry


def _static_document() -> dict[str, Any]:
    """A flow of three tools that all declare static ports -- the shape that must not change."""
    return {
        "name": "seed, pass, sink",
        "steps": [
            {"id": "a", "tool": "seed_rows", "config": {"value": "a"}},
            {"id": "p", "tool": "pass_rows", "inputs": {"rows": "a.rows"}, "config": {"mark": "m"}},
            {"id": "s", "tool": "sink_rows", "inputs": {"rows": "p.rows"}, "config": {}},
        ],
    }


def _two_part_document(
    *,
    inputs: Mapping[str, str] | None = None,
    parts: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """A flow of two seeds feeding one collecting step, with both halves of it overridable."""
    return {
        "name": "two parts",
        "steps": [
            {"id": "a", "tool": "seed_rows", "config": {"value": "a"}},
            {"id": "b", "tool": "seed_rows", "config": {"value": "b"}},
            {
                "id": "fan",
                "tool": "collect_parts",
                "inputs": dict(inputs) if inputs is not None else {"first": "a.rows", "second": "b.rows"},
                "config": {
                    "parts": list(parts)
                    if parts is not None
                    else [{"port": "first", "label": "A"}, {"port": "second", "label": "B"}]
                },
            },
        ],
    }


# ---------------------------------------------------------------------------
# Nothing changes for a tool without the member
# ---------------------------------------------------------------------------


class TestAToolWithoutTheMemberIsUntouched:
    """Twenty-six shipped tools declare static ports. Every one of them behaves as it always did."""

    @pytest.mark.parametrize("tool", REGISTRY.tools(), ids=lambda tool: tool.name)
    def test_a_shipped_tool_that_does_not_derive_its_ports_answers_with_its_attribute(self, tool: Tool) -> None:
        """The helper hands back the static attribute, whatever configuration it is shown."""
        # The excuse is a named list, not a question about the tool. A tool that merely answers to
        # ``input_ports_for`` -- a helper, a property, a mixin -- must fail here rather than be
        # skipped. See ``SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS``.
        if tool.name in SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS:
            pytest.skip(f"{tool.name} derives its ports from its configuration; covered below")

        static = tuple(tool.input_ports)
        assert input_ports_for(tool, {}) == static, "an empty configuration changed the ports"
        assert input_ports_for(tool, {"anything": ["at", "all"]}) == static, "a configuration changed the ports"
        assert catalog_input_ports(tool) == static, "the catalog shows something other than the declared ports"

    def test_the_shipped_tools_still_declare_the_ports_their_own_tests_assert(self) -> None:
        """The static attribute is still a plain tuple of names, read the plain way."""
        assert tuple(REGISTRY.get("filter").input_ports) == ("rows",)
        assert tuple(REGISTRY.get("sort").input_ports) == ("rows",)
        assert tuple(REGISTRY.get("join").input_ports) == ("left", "right")
        assert tuple(REGISTRY.get("read_csv").input_ports) == ()
        assert ports_from_config(REGISTRY.get("join")) is False

    def test_a_tool_without_the_member_is_still_a_tool_and_still_conforms(self) -> None:
        """The protocol did not grow a required member: a tool written yesterday still passes."""
        pass_tool = _PassTool()

        assert isinstance(pass_tool, Tool), "a tool without the member stopped being a Tool"
        assert ports_from_config(pass_tool) is False
        check_tool(pass_tool)
        check_tool(_SeedTool())

    def test_the_boundary_still_refuses_the_same_envelope(self) -> None:
        """An undeclared port and a missing port are refused exactly as before."""
        frame = _seed_frame("a")

        with pytest.raises(PortError) as undeclared:
            run_tool(_PassTool(), Ports({"rows": frame, "extra": frame}), {"mark": "m"})
        assert "extra" in str(undeclared.value) and "rows" in str(undeclared.value)

        with pytest.raises(PortError) as missing:
            run_tool(_PassTool(), Ports(), {"mark": "m"})
        assert "rows" in str(missing.value)

    def test_the_loader_still_accepts_and_refuses_the_same_edges(self) -> None:
        """A static port is wired by name; a port the tool does not have is refused by name."""
        registry = _registry()
        document = _static_document()

        spec = load_pipeline(document, registry=registry)
        assert tuple(spec.step("p").inputs) == ("rows",)
        assert spec.step("p").inputs["rows"] == PortRef("a", "rows")

        document["steps"][1]["inputs"] = {"lines": "a.rows"}
        with pytest.raises(PipelineError) as refusal:
            load_pipeline(document, registry=registry)
        assert "lines" in str(refusal.value) and "rows" in str(refusal.value)

    def test_the_runner_still_gathers_the_static_ports_and_the_source_it_is_handed(self) -> None:
        """A static-port step is fed its one frame, and a source step can still be supplied."""
        spec = load_pipeline(_static_document(), registry=_registry())

        supplied = pl.DataFrame({"id": [9], "seed": ["supplied"]})
        trace = run_pipeline(spec, sources={"a.rows": supplied})

        passed = trace.frames[PortRef("p", "rows")]
        assert passed["seed"].to_list() == ["supplied"], "the supplied source frame did not reach the next step"
        assert passed["mark"].to_list() == ["m"]


# ---------------------------------------------------------------------------
# A tool with the member
# ---------------------------------------------------------------------------


class TestPortsDerivedFromTheConfiguration:
    """The point of the whole change: two steps, one tool, different ports."""

    def test_two_configurations_of_one_tool_have_different_ports(self) -> None:
        """The ports are read off the configuration, in the order the configuration wrote them."""
        tool = _CollectTool()

        one = input_ports_for(tool, {"parts": [{"port": "only", "label": "O"}]})
        three = input_ports_for(
            tool,
            {
                "parts": [
                    {"port": "north", "label": "N"},
                    {"port": "south", "label": "S"},
                    {"port": "east", "label": "E"},
                ]
            },
        )

        assert one == ("only",)
        assert three == ("north", "south", "east"), "the ports lost the order the configuration wrote"
        assert one != three, "two configurations of one tool must be able to have different ports"
        assert tuple(tool.input_ports) == (), "the static attribute is not what answered"
        assert ports_from_config(tool) is True

    def test_a_configuration_naming_nothing_derives_no_ports(self) -> None:
        """Reading nothing is a configuration like any other, not an error."""
        tool = _CollectTool()

        assert input_ports_for(tool, {"parts": []}) == ()

    def test_the_boundary_checks_the_envelope_against_the_derived_ports(self) -> None:
        """A port the configuration did not name is refused, and the refusal lists the ones it did."""
        tool = _CollectTool()
        config = {"parts": [{"port": "first", "label": "A"}]}
        frame = _seed_frame("a")

        result = run_tool(tool, Ports({"first": frame}), config)
        assert result.names == ("rows",)
        assert result["rows"]["part"].to_list() == ["A", "A"]

        with pytest.raises(PortError) as undeclared:
            run_tool(tool, Ports({"first": frame, "second": frame}), config)
        message = str(undeclared.value)
        assert "second" in message, "the refusal does not name the port that was not declared"
        assert "first" in message, "the refusal does not list the port the configuration does declare"

        with pytest.raises(PortError) as missing:
            run_tool(tool, Ports(), config)
        assert "first" in str(missing.value)


class TestTheRunnerHandsOverTheDerivedPorts:
    """The runner gathers the ports the configuration named, keyed by the names it used."""

    def test_execute_is_handed_exactly_the_ports_the_configuration_named(self) -> None:
        """Both frames arrive, under the configuration's own names, from the wired edges."""
        collect = _CollectTool()
        spec = load_pipeline(_two_part_document(), registry=_registry(collect))

        trace = run_pipeline(spec)

        assert collect.executed == [("first", "second")], "the runner did not hand over the derived ports"
        stacked = trace.frames[PortRef("fan", "rows")]
        assert stacked["seed"].to_list() == ["a", "a", "b", "b"], "a frame arrived on the wrong port"
        assert stacked["part"].to_list() == ["A", "A", "B", "B"]

    def test_the_loader_and_the_runner_derive_the_same_ports(self) -> None:
        """Both ask the tool the same question about the same step, and get the same answer.

        Not a claim about when the ports are derived, only that every answer agrees. If the loader
        asked about the configuration as written and the runner asked about it with its defaults
        filled in, a step could load against one set of ports and run against another -- and the
        author would be told nothing until a frame went missing.
        """
        collect = _CollectTool()
        spec = load_pipeline(_two_part_document(), registry=_registry(collect))
        run_pipeline(spec)

        assert collect.derived, "nobody asked the tool for its ports"
        assert set(collect.derived) == {("first", "second")}, f"the answers disagree: {collect.derived}"
        assert collect.executed == [("first", "second")]


class TestTheLoaderValidatesAgainstTheDerivedPorts:
    """A step's edges are checked against the ports its own configuration declares."""

    def test_a_correctly_wired_step_loads_with_the_ports_it_configured(self) -> None:
        """Two parts named, two edges wired, and the step keeps both."""
        spec = load_pipeline(_two_part_document(), registry=_registry())

        assert tuple(spec.step("fan").inputs) == ("first", "second")
        assert spec.step("fan").inputs["second"] == PortRef("b", "rows")

    def test_an_edge_naming_a_port_the_configuration_did_not_declare_is_refused(self) -> None:
        """The message lists the ports this configuration declares, not the static ones."""
        document = _two_part_document(inputs={"first": "a.rows", "third": "b.rows"})

        with pytest.raises(PipelineError) as refusal:
            load_pipeline(document, registry=_registry())

        message = str(refusal.value)
        assert "third" in message, "the refusal does not name the port that does not exist"
        assert "first" in message and "second" in message, (
            "the refusal must list the ports this configuration declares, or the author goes "
            f"looking at the wrong thing: {message}"
        )
        assert "none" not in message, f"the refusal reported the static ports instead: {message}"

    def test_a_declared_port_that_nothing_feeds_is_refused(self) -> None:
        """A part named in the configuration with no edge to it is a hole, and is named as one."""
        document = _two_part_document(inputs={"first": "a.rows"})

        with pytest.raises(PipelineError) as refusal:
            load_pipeline(document, registry=_registry())
        assert "second" in str(refusal.value)


class TestConformanceStillCatchesADuplicatePortName:
    """``check_tool`` has no configuration, so it looks through each worked example's."""

    def test_a_config_derived_tool_whose_examples_are_true_conforms(self) -> None:
        """The ordinary case first: a conforming config-derived tool passes the catalog check."""
        check_tool(_CollectTool())

    def test_an_example_configuration_naming_one_port_twice_is_refused(self) -> None:
        """Two parts on one port is a port collision, and only the example's configuration shows it."""
        collided = WorkedExample(
            scenario="Stack two parts that collide on one port",
            configuration={
                "parts": [
                    {"port": "same", "label": "L"},
                    {"port": "same", "label": "R"},
                ]
            },
            explanation="Both parts name the same port, which cannot carry two frames.",
            inputs=Ports({"same": _seed_frame("a")}),
            outputs=Ports({"rows": _seed_frame("a").with_columns(pl.lit("L").alias("part"))}),
        )

        with pytest.raises(ConformanceError) as refusal:
            check_tool(_CollectTool(examples=(collided,)))

        message = str(refusal.value)
        assert "duplicate input port" in message, f"the refusal does not say what the defect is: {message}"
        assert "same" in message, "the refusal does not name the port that is declared twice"


class TestASourceStepIsStillRecognised:
    """A source reads nothing. The empty static attribute is no longer how that is decided."""

    def test_a_configuration_naming_no_parts_makes_a_source_step(self) -> None:
        """It reads nothing, so a run may hand it its frames, and the tool never runs."""
        collect = _CollectTool()
        document = {
            "name": "supplied",
            "steps": [
                {"id": "fan", "tool": "collect_parts", "config": {"parts": []}},
            ],
        }
        spec = load_pipeline(document, registry=_registry(collect))

        supplied = pl.DataFrame({"id": [7], "seed": ["s"], "part": ["P"]})
        trace = run_pipeline(spec, sources={"fan.rows": supplied})

        assert collect.executed == [], "a supplied source step was run anyway"
        assert trace.frames[PortRef("fan", "rows")]["seed"].to_list() == ["s"]

    def test_a_configuration_naming_parts_is_computed_and_not_a_source(self) -> None:
        """The trap: the static attribute is empty, and this step is still not a source.

        Both halves of the same mistake are here. The runner must run the step rather than wait to
        be handed its output, and ``sources`` must refuse to feed it -- handing a frame to a step
        that computes one would make the flow that ran a different flow from the one in the file.
        """
        collect = _CollectTool()
        spec = load_pipeline(_two_part_document(), registry=_registry(collect))

        run_pipeline(spec)
        assert collect.executed == [("first", "second")], "a computed step was treated as a source"

        with pytest.raises(RunError) as refusal:
            run_pipeline(spec, sources={"fan.rows": _seed_frame("x")})

        message = str(refusal.value)
        assert "computed" in message, f"the refusal does not say why it is not a source: {message}"
        assert "first" in message and "second" in message, (
            f"the refusal does not name the ports that make it computed: {message}"
        )


class TestTheCatalogWithNoConfigurationInHand:
    """The decision: a config-derived tool shows the ports of its first worked example.

    The catalog is generated from the registry with no step in front of it, so it cannot ask a
    config-derived tool what a particular step reads. Three answers were available.

    A static placeholder the tool also declares is unchecked prose. Nothing runs it, so it drifts
    the moment the tool changes, and this library has already decided that a claim nothing checks
    is worse than no claim at all -- that is the whole reason every worked example is executed.

    Text alone -- "the ports come from the configuration" -- is true and useless. The catalog exists
    so a reader can tell what a tool does from the catalog alone. "It depends" fails that on the one
    question the reader came with.

    So the ports come from the first worked example, derived by the very member the runner uses,
    from a configuration that is itself executed and compared every time the catalog is checked.
    They cannot drift. What they can do is mislead, because they are true of that example only, so
    the catalog also carries a flag saying these ports came from a configuration. The page can then
    show the names and say where they came from, which is both halves of what a reader needs.
    """

    def test_a_config_derived_tool_shows_the_ports_of_its_first_worked_example(self) -> None:
        """The names are real and checked, because the example they come from is executed."""
        tool = _CollectTool()
        first = tool.help().examples[0]

        assert catalog_input_ports(tool) == ("left", "right")
        assert catalog_input_ports(tool) == input_ports_for(tool, first.configuration)
        assert ports_from_config(tool) is True

    def test_an_ordinary_tool_shows_the_ports_it_declares(self) -> None:
        """Nothing about the catalog changes for a tool whose ports are fixed."""
        assert catalog_input_ports(_PassTool()) == ("rows",)
        assert catalog_input_ports(_SeedTool()) == ()
        assert ports_from_config(_PassTool()) is False

    def test_the_generated_catalog_shows_those_ports_and_says_where_they_came_from(self) -> None:
        """The page's data carries both halves: the port names, and whether a configuration chose them."""
        spec = importlib.util.spec_from_file_location(
            "library_tests.config_ports_catalog_page", PROJECT_ROOT / "scripts" / "tool_catalog_page.py"
        )
        assert spec is not None and spec.loader is not None, "tool_catalog_page.py is not where the project root says"
        catalog_page = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(catalog_page)

        entries = {entry["name"]: entry for entry in catalog_page.catalog()["tools"]}
        assert entries, "the registry is empty, so this claim would hold vacuously"

        for tool in REGISTRY.tools():
            entry = entries[tool.name]
            assert entry["input_ports"] == list(catalog_input_ports(tool)), f"{tool.name}: wrong ports catalogued"
            assert entry["ports_from_config"] == ports_from_config(tool), (
                f"{tool.name}: the catalog does not say whether a configuration chose these ports"
            )


# ---------------------------------------------------------------------------
# Second round: four holes the first round left open
#
# Each of these pins something the first round asserted somewhere weaker: over worked examples
# rather than over a real step, over a helper rather than over a run, as a skip rather than as an
# assertion, or in a docstring rather than in a check.
# ---------------------------------------------------------------------------


class _StaticAndDerivedTool:
    """Declares a static input port and implements the member as well.

    Nothing forbids this today. The docstring on the protocol says to leave ``input_ports`` empty,
    and a docstring is not a rule. A tool like this lies about what it reads, and every reader of a
    port gets a different answer depending on which one it asked.
    """

    name = "static_and_derived"
    description = "Hand the first frame on, and claim two different sets of ports while doing it."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]:
        return tuple(str(part["port"]) for part in config.get("parts", ()))

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "parts": ConfigField(
                type="array",
                description="One entry per input port this step reads.",
                items=ConfigField(
                    type="object",
                    description="One named part.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port this part arrives on.",
                            required=True,
                            non_empty=True,
                        )
                    },
                ),
            )
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description="Hands the first frame it was given straight back out.",
            examples=(
                WorkedExample(
                    scenario="Pass one frame through",
                    configuration={"parts": [{"port": "rows"}]},
                    explanation="One part named, one frame in, the same frame out.",
                    inputs=Ports({"rows": _seed_frame("a")}),
                    outputs=Ports({"rows": _seed_frame("a")}),
                    check_row_order=True,
                ),
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        if not inputs.names:
            return Ports({"rows": pl.DataFrame(schema={"id": pl.Int64, "seed": pl.String})})
        return Ports({"rows": inputs[inputs.names[0]]})


class _DefaultingCollectTool:
    """Same idea as the collecting tool, but its port field carries a template default.

    This is the tool that can tell whether derivation saw the file or saw the file with its blanks
    filled in. A step that writes no parts reads nothing. A step derived from the filled-in
    configuration would instead read one port named ``fallback``, which nobody wrote anywhere.
    """

    name = "collect_defaulted_parts"
    description = "Stack the frames on the named parts, with a part named by default."
    category = "output"
    input_ports: Sequence[str] = ()
    output_ports = ("rows",)

    def __init__(self) -> None:
        #: Every port tuple this tool was asked to derive, in the order it was asked.
        self.derived: list[tuple[str, ...]] = []
        #: The port names actually handed to ``execute``, per call.
        self.executed: list[tuple[str, ...]] = []
        #: The configuration ``execute`` was handed, per call. Defaults are filled in by then.
        self.executed_with: list[Mapping[str, Any]] = []

    def input_ports_for(self, config: Mapping[str, Any]) -> Sequence[str]:
        ports = tuple(str(part["port"]) for part in config.get("parts", ()))
        self.derived.append(ports)
        return ports

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "parts": ConfigField(
                type="array",
                description="One entry per input port this step reads, in the order they are read.",
                default=[{"port": "fallback", "label": "F"}],
                items=ConfigField(
                    type="object",
                    description="One named part.",
                    fields={
                        "port": ConfigField(
                            type="string",
                            description="The input port this part arrives on.",
                            required=True,
                            non_empty=True,
                        ),
                        "label": ConfigField(
                            type="string",
                            description="What the part column says for this part's rows.",
                            required=True,
                            non_empty=True,
                        ),
                    },
                ),
            )
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Every part named in the configuration arrives on its own input port. The port "
                "field has a default, so a step that writes nothing is a step that reads nothing."
            ),
            examples=(
                WorkedExample(
                    scenario="Stack one named part",
                    configuration={"parts": [{"port": "left", "label": "L"}]},
                    explanation="One part named, so one input port, carrying a column saying which part it is.",
                    inputs=Ports({"left": _seed_frame("a")}),
                    outputs=Ports({"rows": _seed_frame("a").with_columns(pl.lit("L").alias("part"))}),
                    check_row_order=True,
                ),
            ),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self.executed.append(inputs.names)
        self.executed_with.append(dict(config))
        if not inputs.names:
            return Ports({"rows": pl.DataFrame(schema={"id": pl.Int64, "seed": pl.String, "part": pl.String})})
        labels = {str(part["port"]): str(part["label"]) for part in config.get("parts", ())}
        stacked = pl.concat(
            [inputs[name].with_columns(pl.lit(labels.get(name, name)).alias("part")) for name in inputs.names]
        )
        return Ports({"rows": stacked})


def _two_steps_of_one_tool_document() -> dict[str, Any]:
    """Two steps of the same config-derived tool, reading different ports.

    The first step reads two ports, the second reads one. No static attribute could say that, and
    no answer cached against the tool could either.
    """
    return {
        "name": "two steps, one tool",
        "steps": [
            {"id": "a", "tool": "seed_rows", "config": {"value": "a"}},
            {"id": "b", "tool": "seed_rows", "config": {"value": "b"}},
            {
                "id": "fan",
                "tool": "collect_parts",
                "inputs": {"first": "a.rows", "second": "b.rows"},
                "config": {"parts": [{"port": "first", "label": "A"}, {"port": "second", "label": "B"}]},
            },
            {
                "id": "solo",
                "tool": "collect_parts",
                "inputs": {"only": "fan.rows"},
                "config": {"parts": [{"port": "only", "label": "S"}]},
            },
        ],
    }


def _duplicate_port_document(*, wired: bool) -> dict[str, Any]:
    """One seed feeding one collecting step whose configuration names the same port twice.

    One seed, not two, so the duplicate is the only thing wrong with the file. Two seeds would
    leave one of them feeding nothing, and the loader would refuse that instead.
    """
    return {
        "name": "one port, twice",
        "steps": [
            {"id": "a", "tool": "seed_rows", "config": {"value": "a"}},
            {
                "id": "fan",
                "tool": "collect_parts",
                "inputs": {"same": "a.rows"} if wired else {},
                "config": {"parts": [{"port": "same", "label": "L"}, {"port": "same", "label": "R"}]},
            },
        ],
    }


class TestAStepNamingOnePortTwiceIsRefused:
    """A configuration naming one port twice is refused when the file is loaded.

    The first round checked this over worked examples only, inside ``check_tool``. An authored step
    was never checked at all. Neither the loader nor the boundary catches it: both compare sets of
    names, and a repeated name disappears into a set. So the step loads, the runner writes the same
    frame into the same dict key twice, and ``execute`` is handed one port where the configuration
    declared two. For a workbook writer that is a tab missing from the file, or one frame written
    to two tabs, with nothing said anywhere.

    The refusal belongs in the loader. The defect is in the file, not in the data: the same file is
    wrong every time it is read, whatever frames are flowing. That is where every other "this file
    is wrong" refusal already lives, and the author is told once, at load, before anything runs.
    The boundary cannot be the only place, because by the time the envelope is built one of the two
    frames is already gone.
    """

    def test_a_step_whose_configuration_names_one_port_twice_is_refused_at_load(self) -> None:
        """The message names the repeated port, and the tool never runs."""
        collect = _CollectTool()

        with pytest.raises(PipelineError) as refusal:
            load_pipeline(_duplicate_port_document(wired=True), registry=_registry(collect))

        message = str(refusal.value)
        assert "duplicate input port" in message, f"the refusal does not say what the defect is: {message}"
        assert "same" in message, f"the refusal does not name the port that is declared twice: {message}"
        assert "fan" in message, f"the refusal does not name the step to go and fix: {message}"
        assert collect.executed == [], "the step ran despite naming one port twice"

    def test_the_duplicate_is_reported_ahead_of_the_unfed_port_it_causes(self) -> None:
        """With no edges written, the author is told about the duplicate, not about a hole.

        A repeated port with nothing wired to it trips the "does not feed input port" refusal too.
        That one sends the author to the arrows, when the thing to fix is the configuration.
        """
        with pytest.raises(PipelineError) as refusal:
            load_pipeline(_duplicate_port_document(wired=False), registry=_registry())

        message = str(refusal.value)
        assert "duplicate input port" in message, f"the author was sent to the wrong defect: {message}"
        assert "same" in message


class TestTwoStepsOfOneConfigDerivedTool:
    """Two steps of one tool, with different ports, in one document, run end to end.

    Every other document in this file has a single config-derived step. That leaves room for the
    ports to be worked out once per tool rather than once per step. A cache keyed on the tool would
    pass every test written so far, and hand the first step's frames to the second. That is the
    exact failure this feature exists to prevent, so it is pinned against a real run.
    """

    def test_the_loader_wires_each_step_against_its_own_ports(self) -> None:
        """One step keeps two edges, the other keeps one, and both are of the same tool."""
        spec = load_pipeline(_two_steps_of_one_tool_document(), registry=_registry())

        assert tuple(spec.step("fan").inputs) == ("first", "second")
        assert tuple(spec.step("solo").inputs) == ("only",), "the second step was wired against the first step's ports"
        assert spec.step("solo").inputs["only"] == PortRef("fan", "rows")

    def test_each_step_is_handed_the_frames_its_own_configuration_named(self) -> None:
        """Two ports to the first step, one to the second, and the right rows in each."""
        collect = _CollectTool()
        spec = load_pipeline(_two_steps_of_one_tool_document(), registry=_registry(collect))

        trace = run_pipeline(spec)

        assert collect.executed == [("first", "second"), ("only",)], (
            f"the two steps were not handed their own ports: {collect.executed}"
        )
        assert ("only",) in set(collect.derived), f"nobody asked the tool what the second step reads: {collect.derived}"

        stacked = trace.frames[PortRef("fan", "rows")]
        assert stacked["seed"].to_list() == ["a", "a", "b", "b"]

        onward = trace.frames[PortRef("solo", "rows")]
        assert onward["seed"].to_list() == ["a", "a", "b", "b"], "the second step read the wrong frame"
        assert onward["part"].to_list() == ["S", "S", "S", "S"], "the second step ran on the first step's configuration"


#: Shipped tools that work their input ports out from a step's configuration.
#:
#: A tool joining this set has to be an edit somebody reviews. That is the point. The
#: membership test is ``isinstance`` against a runtime-checkable protocol, which only asks whether
#: a member of that name is present. A helper, a cached property or a mixin called
#: ``input_ports_for`` would quietly join this set, and a guard that skipped on membership would
#: excuse the very tool it exists to check.
#:
#: ``write_excel`` is the first member: it reads one port per tab its ``sheets`` name, so two
#: steps of it read different frames and no static attribute can say what either one reads. It has
#: its own coverage in ``tests/tools/test_write_excel.py``.
#:
#: ``publish_tables`` is the second, for the same shape and the same reason: it reads one port per
#: table its ``tables`` name, and a publish of five tables reads five frames where a publish of one
#: reads one. Its coverage is in ``tests/tools/test_publish_tables.py``.
SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS: frozenset[str] = frozenset({"write_excel", "publish_tables"})


class TestTheShippedCatalogIsCheckedNotExcused:
    """No shipped tool derives its ports, and that is asserted rather than assumed."""

    def test_only_the_listed_shipped_tools_derive_their_ports(self) -> None:
        """The set of config-deriving shipped tools is the list above, exactly."""
        deriving = {tool.name for tool in REGISTRY.tools() if ports_from_config(tool)}

        assert deriving == set(SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS), (
            "a shipped tool now answers to input_ports_for. If that is deliberate, add it to "
            "SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS and give it its own coverage. If it is an "
            "accident -- a helper or a property that happens to carry that name -- rename it, "
            f"because every reader of a port now asks it what a step reads: {sorted(deriving)}"
        )

    def test_every_shipped_tool_answers_with_its_attribute(self) -> None:
        """Whatever configuration a shipped tool is shown, its ports do not move."""
        for tool in REGISTRY.tools():
            if tool.name in SHIPPED_TOOLS_THAT_DERIVE_THEIR_PORTS:
                continue
            static = tuple(tool.input_ports)
            assert input_ports_for(tool, {"parts": [{"port": "invented"}]}) == static, (
                f"{tool.name}: a configuration changed the ports of a tool that declares them"
            )


class TestAToolMayNotDeclareStaticPortsAndDeriveThem:
    """A tool that declares static ports and implements the member is refused by the check.

    Until now this was a sentence in a docstring. A convention an author has to remember is not a
    rule, and this one is load-bearing. With such a tool, configured to derive no ports, the run
    guard asks with the configuration, gets nothing, and accepts a supplied frame. The decision to
    skip the step is taken from the static attribute, which says the step reads one port. So the
    step runs, the supplied frame is dropped, and the flow that ran is not the flow the caller
    asked for. Nothing is said.

    Making it a conformance error is the fix that holds without anybody remembering it.
    """

    def test_a_tool_that_does_both_is_refused_by_the_conformance_check(self) -> None:
        """The message names the member and the static ports it contradicts."""
        with pytest.raises(ConformanceError) as refusal:
            check_tool(_StaticAndDerivedTool())

        message = str(refusal.value)
        assert "static_and_derived" in message, f"the refusal does not name the tool: {message}"
        assert "input_ports_for" in message, f"the refusal does not name the member: {message}"
        assert "rows" in message, f"the refusal does not name the static ports it contradicts: {message}"

    def test_a_tool_that_does_only_one_of_them_still_conforms(self) -> None:
        """The rule refuses the pair, not either half of it."""
        check_tool(_CollectTool())
        check_tool(_PassTool())
        check_tool(_SeedTool())


class TestTheUnfedPortRefusalListsTheDerivedPorts:
    """The hole refusal lists the ports this configuration declares, like its sibling does."""

    def test_the_refusal_lists_every_port_this_configuration_declares(self) -> None:
        """Not only the port with no edge: the whole list, so the author can see what is missing."""
        document = _two_part_document(inputs={"first": "a.rows"})

        with pytest.raises(PipelineError) as refusal:
            load_pipeline(document, registry=_registry())

        message = str(refusal.value)
        assert "second" in message, "the refusal does not name the port nothing feeds"
        assert "first" in message, f"the refusal does not list the ports this configuration declares: {message}"
        assert "none" not in message, f"the refusal reported the static ports instead: {message}"


class TestDerivationReadsTheConfigurationAsAuthored:
    """Ports come from the file a person wrote, never from the file with its blanks filled in.

    True today, but only because the reference tool's port field has no default, so no ordering
    could show. A tool whose port field carries one makes the difference visible. A step writing
    nothing reads nothing, where a defaulted configuration would have it read a port named in the
    template and nowhere else.
    """

    def _registry_with_defaults(self, tool: _DefaultingCollectTool) -> ToolRegistry:
        registry = _registry()
        registry.register(tool)
        return registry

    def test_a_step_writing_no_parts_derives_no_ports_at_the_loader_and_the_runner(self) -> None:
        """The template's default is never a port. The step reads nothing, and is a source."""
        tool = _DefaultingCollectTool()
        document = {
            "name": "nothing written",
            "steps": [{"id": "fan", "tool": "collect_defaulted_parts", "config": {}}],
        }

        spec = load_pipeline(document, registry=self._registry_with_defaults(tool))
        assert tuple(spec.step("fan").inputs) == (), "the loader wired a port that only the template names"

        run_pipeline(spec)

        assert tool.executed == [()], f"the runner gathered a port nobody wrote: {tool.executed}"
        assert set(tool.derived) == {()}, f"someone derived ports from a defaulted configuration: {tool.derived}"
        assert tool.executed_with and "parts" in tool.executed_with[0], (
            "execute was not handed the filled-in configuration; only derivation reads it as authored"
        )

    def test_the_same_tool_derives_the_ports_a_step_does_write(self) -> None:
        """The other half: a step that writes its parts reads exactly those."""
        tool = _DefaultingCollectTool()
        document = {
            "name": "parts written",
            "steps": [
                {"id": "a", "tool": "seed_rows", "config": {"value": "a"}},
                {
                    "id": "fan",
                    "tool": "collect_defaulted_parts",
                    "inputs": {"here": "a.rows"},
                    "config": {"parts": [{"port": "here", "label": "H"}]},
                },
            ],
        }

        spec = load_pipeline(document, registry=self._registry_with_defaults(tool))
        trace = run_pipeline(spec)

        assert tuple(spec.step("fan").inputs) == ("here",)
        assert tool.executed == [("here",)]
        assert trace.frames[PortRef("fan", "rows")]["part"].to_list() == ["H", "H"]
