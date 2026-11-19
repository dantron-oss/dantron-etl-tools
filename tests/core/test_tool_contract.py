"""Tests for the tool contract: the named-frame envelope, the tool anatomy, and the registry.

Three things are specified here, and each one exists to stop a different class of defect.

**Ports** is the envelope. A tool takes named frames in and hands named frames out, because a tool
in this library may have *several outputs*: a filter emits the rows that matched **and** the rows
that did not, a match emits the pairs plus the leftovers from each side. Those side outputs are the
whole point -- rows a step discards stop being something a reader reconstructs afterwards with an
anti-join and become a first-class output a flow can route to a report, to an assertion, or to
nothing at all. A bare ``dict`` cannot carry that: a tool that forgets one of its declared outputs
would hand a ``KeyError`` to whatever runs next, three steps away from the mistake. So the envelope
is checked at the boundary of every call, and every declared output is present on every call --
empty but carrying its columns when nothing reached it, so that a downstream predicate validated
against it does not fail for the wrong reason.

**The tool anatomy** is deliberately close to the one proven in the library this replaces: machine
name, one-line description, category, detailed help with worked examples, a typed configuration
template, config validation, execute. Two things change. Ports replace "the right-hand side is a
file path in the config", which made a join impossible to compose. And a config field can declare
itself an *expression*, which is the seam to the expression language: a loader can lift the column
references out of a predicate and check them against the upstream frame before a row is read.

**The registry** is a catalog, and the conformance test below is what makes it trustworthy: every
worked example in it is executed. An example that has drifted from its tool is worse than no
example at all, because somebody will copy it. The conformance test is parametrised over whatever
the registry holds, so it passes on an empty registry today and covers every tool added later
without anybody remembering to add it to a list.

Vocabulary here is deliberately generic -- code, amount, key, label -- because these fixtures are
written to be liftable into the library's own test suite when it is extracted, and the library must
carry no caller's nouns.
"""

import ast
import inspect
import os
import re
import shutil
import textwrap
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest
import xlsxwriter

from dantron_etl import authoring as tool_contract
from dantron_etl.expr import ExpressionError, compile_expr, references
from dantron_etl.authoring import (
    ConfigError,
    ConfigField,
    ConformanceError,
    DataError,
    ExecutionError,
    FIELD_TYPES,
    MAX_REPORTED_ROWS,
    PortError,
    Ports,
    REGISTRY,
    REPLACE_FIELD,
    RegistryError,
    Tool,
    ToolError,
    ToolHelp,
    ToolRegistry,
    WorkedExample,
    YAML_UNSAFE_VALUES,
    check_output_column,
    check_registry,
    check_tool,
    compile_predicate_for,
    compiled_dtype,
    require_columns,
    run_tool,
    temporary_environment,
)
from dantron_etl.config import apply_defaults, expression_fields, validate_against_template

# ---------------------------------------------------------------------------
# Reference tools
#
# Two tools, written against the contract and nothing else, because the contract cannot be
# exercised without something that conforms to it. They are the two shapes that motivate the
# envelope: one input and two outputs, and two inputs and three outputs.
# ---------------------------------------------------------------------------


class _KeepRowsTool:
    """One input, two outputs: the rows a predicate keeps, and the rows it does not."""

    name = "keep_rows"
    description = "Split rows into those matching a predicate and those not matching it."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("matched", "unmatched")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "predicate": ConfigField(
                type="expression",
                description="Condition a row must satisfy to reach the matched output.",
                required=True,
            ),
            "null_rows": ConfigField(
                type="string",
                description="Where a row whose predicate is null goes.",
                required=False,
                default="exclude",
                enum=["exclude", "include"],
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Evaluates the predicate against every row and routes the row to one of two "
                "outputs. Nothing is dropped: a row that does not match leaves on the unmatched "
                "output rather than disappearing."
            ),
            examples=(
                WorkedExample(
                    scenario="Keep the positive amounts and set the rest aside",
                    configuration={"predicate": "[amount] > 0"},
                    explanation=(
                        "Two rows have a positive amount and leave on matched; the negative one "
                        "leaves on unmatched, where a later step can report on it."
                    ),
                    inputs=Ports({"rows": pl.DataFrame({"code": ["A", "B", None], "amount": [10.0, -5.0, 3.0]})}),
                    outputs=Ports(
                        {
                            "matched": pl.DataFrame({"code": ["A", None], "amount": [10.0, 3.0]}),
                            "unmatched": pl.DataFrame({"code": ["B"], "amount": [-5.0]}),
                        }
                    ),
                ),
                WorkedExample(
                    scenario="Treat a null amount as a match",
                    configuration={"predicate": "[amount] > 0", "null_rows": "include"},
                    explanation=(
                        "A comparison against null is null, not false, so a null row would "
                        "ordinarily leave on unmatched. Asking for null_rows=include sends it to "
                        "matched instead, and the unmatched output is empty but still has its "
                        "amount column."
                    ),
                    inputs=Ports({"rows": pl.DataFrame({"amount": [1.0, None]})}),
                    outputs=Ports(
                        {
                            "matched": pl.DataFrame({"amount": [1.0, None]}),
                            "unmatched": pl.DataFrame({"amount": []}, schema={"amount": pl.Float64}),
                        }
                    ),
                ),
            ),
            notes=(
                "A row whose predicate is null does not match, following the engine rather than "
                "intuition; null_rows=include overrides that.",
                "Both outputs are always produced, and an empty one keeps the input's columns.",
            ),
            field_help={"predicate": "A condition over the input columns, e.g. [amount] > 0."},
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_against_template(self.config_template(), config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to add beyond the template -- what a tool with no config/schema rule writes."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        keep_nulls = config["null_rows"] == "include"
        mask = compile_expr(config["predicate"], frame.columns).fill_null(keep_nulls)
        return Ports({"matched": frame.filter(mask), "unmatched": frame.filter(~mask)})


class _MatchRowsTool:
    """Two inputs, three outputs: the pairs, and the leftovers from each side."""

    name = "match_rows"
    description = "Pair rows from two inputs on key columns, keeping both sets of leftovers."
    category = "combine"
    input_ports = ("left", "right")
    output_ports = ("matched", "left_only", "right_only")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "keys": ConfigField(
                type="array",
                description="Key columns present under the same name on both inputs.",
                required=False,
            ),
            "left_on": ConfigField(
                type="array",
                description="Key columns on the left input, when the names differ.",
                required=False,
            ),
            "right_on": ConfigField(
                type="array",
                description="Key columns on the right input, when the names differ.",
                required=False,
            ),
            "suffix": ConfigField(
                type="string",
                description="Suffix added to a right-hand column whose name collides.",
                required=False,
                default="_right",
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Pairs rows across two inputs and reports what did not pair. The unpaired rows "
                "leave on their own outputs, so nothing has to be reconstructed later."
            ),
            examples=(
                WorkedExample(
                    scenario="Pair on a shared key column",
                    configuration={"keys": ["key"]},
                    explanation=(
                        "One key is on both sides and leaves on matched. The key only on the left "
                        "leaves on left_only, and the key only on the right on right_only."
                    ),
                    inputs=Ports(
                        {
                            "left": pl.DataFrame({"key": [1, 2], "lv": ["a", "b"]}),
                            "right": pl.DataFrame({"key": [2, 3], "rv": ["x", "y"]}),
                        }
                    ),
                    outputs=Ports(
                        {
                            "matched": pl.DataFrame({"key": [2], "lv": ["b"], "rv": ["x"]}),
                            "left_only": pl.DataFrame({"key": [1], "lv": ["a"]}),
                            "right_only": pl.DataFrame({"key": [3], "rv": ["y"]}),
                        }
                    ),
                ),
            ),
            notes=("Give either keys, or both of left_on and right_on -- never both forms at once.",),
            field_help={"keys": "Use this when the key column has the same name on both inputs."},
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_against_template(self.config_template(), config)
        shared = config.get("keys")
        split = config.get("left_on"), config.get("right_on")
        if shared and any(split):
            raise ConfigError("give either 'keys' or both of 'left_on' and 'right_on', not both forms")
        if not shared and not all(split):
            raise ConfigError("give either 'keys' or both of 'left_on' and 'right_on'")

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to add beyond the template -- what a tool with no config/schema rule writes."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        left, right = inputs["left"], inputs["right"]
        shared = config.get("keys")
        if shared:
            keys: dict[str, Any] = {"on": list(shared)}
            flipped: dict[str, Any] = {"on": list(shared)}
        else:
            keys = {"left_on": list(config["left_on"]), "right_on": list(config["right_on"])}
            flipped = {"left_on": list(config["right_on"]), "right_on": list(config["left_on"])}
        return Ports(
            {
                "matched": left.join(right, how="inner", suffix=config["suffix"], **keys),
                "left_only": left.join(right, how="anti", **keys),
                "right_only": right.join(left, how="anti", **flipped),
            }
        )


# ---------------------------------------------------------------------------
# A configurable stub, so a broken tool is one constructor call
# ---------------------------------------------------------------------------


def _label_rows(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Add the configured label as a column -- the stub's conforming behaviour."""
    return Ports({"out": inputs["in"].with_columns(pl.lit(config["label"]).alias("label"))})


class _StubTool:
    """A conforming tool whose every declared part can be replaced with a broken one."""

    def __init__(
        self,
        *,
        name: str = "stub",
        description: str = "Add a fixed label column to every row.",
        category: str = "transform",
        input_ports: Sequence[str] = ("in",),
        output_ports: Sequence[str] = ("out",),
        template: Mapping[str, ConfigField] | None = None,
        detailed_description: str = "Adds one column holding the configured label to every row.",
        examples: Sequence[WorkedExample] | None = None,
        field_help: Mapping[str, str] | None = None,
        execute_fn: Callable[[Ports, Mapping[str, Any]], Ports] = _label_rows,
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.input_ports = tuple(input_ports)
        self.output_ports = tuple(output_ports)
        self._template = (
            dict(template)
            if template is not None
            else {"label": ConfigField(type="string", description="The label to add.", required=True)}
        )
        self._detailed_description = detailed_description
        self._examples = (
            tuple(examples)
            if examples is not None
            else (
                WorkedExample(
                    scenario="Label every row",
                    configuration={"label": "x"},
                    explanation="Every row gains a label column holding 'x'.",
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"out": pl.DataFrame({"id": [1], "label": ["x"]})}),
                ),
            )
        )
        self._field_help = dict(field_help) if field_help is not None else {}
        self._execute_fn = execute_fn

    def config_template(self) -> Mapping[str, ConfigField]:
        return dict(self._template)

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=self._detailed_description,
            examples=self._examples,
            notes=("A stub used to exercise the contract.",),
            field_help=dict(self._field_help),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_against_template(self.config_template(), config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to add beyond the template -- what a tool with no config/schema rule writes."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return self._execute_fn(inputs, config)


def _frame() -> pl.DataFrame:
    """A one-row frame to feed a stub."""
    return pl.DataFrame({"id": [1]})


CONFORMANCE_SUBJECTS: tuple[Any, ...] = (*REGISTRY.tools(), _KeepRowsTool(), _MatchRowsTool(), _StubTool())


# ---------------------------------------------------------------------------
# Ports -- the envelope
# ---------------------------------------------------------------------------


class TestPorts:
    """A port is a named frame, and the set of them is ordered, immutable and typed."""

    def test_ports_hold_named_frames_in_a_deterministic_order(self) -> None:
        """Names, items and iteration all follow construction order, on every pass."""
        matched, unmatched = pl.DataFrame({"id": [1]}), pl.DataFrame({"id": [2]})
        ports = Ports({"matched": matched, "unmatched": unmatched})

        assert ports.names == ("matched", "unmatched")
        assert tuple(ports) == ("matched", "unmatched")
        assert tuple(ports) == tuple(ports), "iterating twice must give the same order"
        assert tuple(name for name, _ in ports.items()) == ("matched", "unmatched")
        assert len(ports) == 2
        assert "matched" in ports and "missing" not in ports
        assert ports["unmatched"].equals(unmatched)

        with pytest.raises(PortError):
            Ports({"rows": [1, 2, 3]})

    def test_ports_are_immutable_once_built(self) -> None:
        """No item assignment, no attribute assignment, and no leak from the source mapping."""
        source = {"rows": pl.DataFrame({"id": [1]})}
        ports = Ports(source)

        with pytest.raises(TypeError):
            ports["rows"] = pl.DataFrame({"id": [9]})
        with pytest.raises(AttributeError):
            ports.names = ("other",)

        source["rows"] = pl.DataFrame({"id": [9]})
        source["extra"] = pl.DataFrame({"id": [9]})
        assert ports.names == ("rows",)
        assert ports["rows"]["id"].to_list() == [1]

    def test_an_unknown_port_lookup_is_refused_by_name(self) -> None:
        """Asking for a port that is not there names it and lists what is, rather than KeyError."""
        ports = Ports({"matched": pl.DataFrame({"id": [1]}), "unmatched": pl.DataFrame({"id": [2]})})

        with pytest.raises(PortError) as caught:
            ports["mateched"]

        message = str(caught.value)
        assert "mateched" in message
        assert "matched" in message and "unmatched" in message
        assert isinstance(caught.value, ToolError)

    def test_an_empty_output_port_keeps_the_schema_of_its_input(self) -> None:
        """A port nothing reached is empty but typed, so a downstream check does not misfire."""
        frame = pl.DataFrame({"code": ["A", "B"], "amount": [10.0, 3.0]})
        result = run_tool(_KeepRowsTool(), Ports({"rows": frame}), {"predicate": "[amount] < 0"})

        assert result.names == ("matched", "unmatched")
        assert result["matched"].height == 0, "no row satisfies the predicate"
        assert result["matched"].schema == frame.schema, "an empty port still carries its columns"
        assert result["unmatched"].height == 2

        # The reason it matters: the next step validates its predicate against the empty port's
        # columns, and must not be told the column is missing.
        compile_expr("[amount] > 0", result["matched"].columns)


# ---------------------------------------------------------------------------
# The tool boundary
# ---------------------------------------------------------------------------


def _port_violation_cases() -> list[tuple[Any, Ports, Mapping[str, Any], tuple[str, ...]]]:
    """(tool, inputs, config, substrings the refusal must contain) for each way to break ports."""
    label = {"label": "x"}
    return [
        (
            _StubTool(output_ports=("out", "rejected")),
            Ports({"in": _frame()}),
            label,
            ("rejected", "out"),
        ),
        (
            _StubTool(execute_fn=lambda inputs, config: Ports({"out": inputs["in"], "surprise": inputs["in"]})),
            Ports({"in": _frame()}),
            label,
            ("surprise", "out"),
        ),
        (
            _StubTool(),
            Ports({"in": _frame(), "other": _frame()}),
            label,
            ("other", "in"),
        ),
        (
            _StubTool(),
            Ports({}),
            label,
            ("in",),
        ),
        (
            _StubTool(execute_fn=lambda inputs, config: Ports({"out": pl.DataFrame()})),
            Ports({"in": _frame()}),
            label,
            ("out",),
        ),
    ]


class TestToolBoundary:
    """Every call goes through one boundary, and the boundary is where the envelope is checked."""

    @pytest.mark.parametrize(
        ("tool", "inputs", "config", "expected_substrings"),
        _port_violation_cases(),
        ids=[
            "declared_output_never_produced",
            "undeclared_output_produced",
            "undeclared_input_supplied",
            "declared_input_missing",
            "output_port_without_columns",
        ],
    )
    def test_port_declaration_violations_fail_at_the_boundary(
        self,
        tool: Any,
        inputs: Ports,
        config: Mapping[str, Any],
        expected_substrings: tuple[str, ...],
    ) -> None:
        """A port that is missing, undeclared or untyped is a named error, not a later KeyError."""
        with pytest.raises(PortError) as caught:
            run_tool(tool, inputs, config)

        message = str(caught.value)
        for substring in expected_substrings:
            assert substring in message, f"refusal should name {substring!r}: {message}"

    def test_the_boundary_validates_config_fills_defaults_and_orders_outputs(self) -> None:
        """Config is checked before execute runs, defaults are present in it, and ports come back
        in declaration order regardless of the order the tool built them."""
        seen: list[Mapping[str, Any]] = []

        def spy(inputs: Ports, config: Mapping[str, Any]) -> Ports:
            seen.append(dict(config))
            return Ports({"out": inputs["in"], "spare": inputs["in"]})

        template = {
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "mode": ConfigField(type="string", description="How to add it.", required=False, default="append"),
        }
        tool = _StubTool(output_ports=("spare", "out"), template=template, execute_fn=spy, examples=())

        with pytest.raises(ConfigError):
            run_tool(tool, Ports({"in": _frame()}), {"mode": "append"})
        assert seen == [], "execute must not run on a config that does not validate"

        result = run_tool(tool, Ports({"in": _frame()}), {"label": "x"})
        assert seen == [{"label": "x", "mode": "append"}], "a default must be filled in before execute"
        assert result.names == tool.output_ports


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """Validation derived from the template, plus a hook for what a template cannot express."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"null_rows": "exclude"}, id="required_field_missing"),
            pytest.param({"predicate": "[amount] > 0", "predicat": "[amount] > 0"}, id="unknown_field"),
            pytest.param({"predicate": ["amount", ">", 0]}, id="wrong_type_for_field"),
            pytest.param({"predicate": "[amount] > 0", "null_rows": "keep"}, id="value_outside_enum"),
            pytest.param({"predicate": "[amount] >"}, id="expression_that_does_not_compile"),
            pytest.param({"predicate": "[type] = '${rules.category}'"}, id="unresolved_parameter_placeholder"),
        ],
    )
    def test_template_derived_rules_refuse_a_bad_config(self, config: Mapping[str, Any]) -> None:
        """Everything the template can express is enforced from the template, uniformly.

        The last case is the boundary with the loader: a value may be written as a reference to a
        pipeline parameter, and resolving it is the loader's job, not the tool's. So a tool that is
        handed an unresolved placeholder must refuse it loudly. The alternative -- passing it
        through -- means a predicate quietly matches nothing and the run looks successful.
        """
        with pytest.raises(ConfigError):
            _KeepRowsTool().validate_config(config)

    def test_cross_field_rules_run_in_addition_to_template_rules(self) -> None:
        """A hand-written rule expresses what a per-field template cannot, without replacing it."""
        tool = _MatchRowsTool()

        tool.validate_config({"keys": ["key"]})
        tool.validate_config({"left_on": ["a"], "right_on": ["b"]})

        with pytest.raises(ConfigError):
            tool.validate_config({"keys": ["key"], "left_on": ["a"], "right_on": ["b"]})
        with pytest.raises(ConfigError):
            tool.validate_config({"left_on": ["a"]})
        with pytest.raises(ConfigError):
            tool.validate_config({"keys": ["key"], "unknown": 1})


# ---------------------------------------------------------------------------
# The seam to the expression language
# ---------------------------------------------------------------------------


class TestExpressionFieldSeam:
    """A config field says it holds an expression, so a loader can check it before execution."""

    def test_expression_fields_let_a_loader_preflight_column_references(self) -> None:
        """The declared field type is the whole seam: find the field, read its column references,
        check them against the upstream frame, and fail on a typo before a row is read."""
        assert "expression" in FIELD_TYPES

        template = _KeepRowsTool().config_template()
        assert expression_fields(template) == ("predicate",)

        upstream = pl.DataFrame({"code": ["A"], "amount": [1.0]})
        config = {"predicate": "[amount] > 0 and [code] = 'A'"}
        assert references(config["predicate"]) == frozenset({"amount", "code"})
        assert references(config["predicate"]) <= set(upstream.columns)
        compile_expr(config["predicate"], upstream.columns)

        with pytest.raises(ExpressionError):
            compile_expr("[amuont] > 0", upstream.columns)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    """Tools are resolved by name, and a name collision is refused rather than silently winning."""

    def test_registry_resolves_by_name_and_refuses_duplicates_and_strangers(self) -> None:
        """Register, look up, list; a repeat name and an unknown name are both named errors."""
        registry = ToolRegistry()
        keep, match = _KeepRowsTool(), _MatchRowsTool()
        registry.register(match)
        registry.register(keep)

        assert registry.get("keep_rows") is keep
        assert registry.names() == ("keep_rows", "match_rows"), "listing must be deterministic"
        assert registry.tools() == (keep, match)
        assert len(registry) == 2
        assert "match_rows" in registry

        with pytest.raises(RegistryError) as duplicate:
            registry.register(_KeepRowsTool())
        assert "keep_rows" in str(duplicate.value)

        with pytest.raises(RegistryError) as unknown:
            registry.get("keep_row")
        assert "keep_row" in str(unknown.value) and "keep_rows" in str(unknown.value)

        assert isinstance(keep, Tool)
        with pytest.raises(RegistryError):
            registry.register(object())


# ---------------------------------------------------------------------------
# Catalog conformance
# ---------------------------------------------------------------------------


def _broken_tools() -> list[tuple[Any, tuple[str, ...], tuple[str, ...]]]:
    """One tool per conformance rule, each broken in exactly that one way.

    Each case carries the substrings its refusal must contain and the substrings it must not, so a
    tool broken in way X cannot pass this test by being refused for unrelated reason Y. The
    forbidden substrings are what separate the two example failures below: an example whose
    *configuration* is wrong and an example that uses a *port the tool does not declare* are two
    different defects with two different fixes, and a single "does not run" message for both tells
    a reader neither.
    """
    conforming_example = _StubTool().help().examples[0]
    return [
        (
            _StubTool(
                execute_fn=lambda inputs, config: Ports({"out": inputs["in"].with_columns(pl.lit("y").alias("label"))})
            ),
            ("drifted", "out"),
            (),
        ),
        (
            _StubTool(
                examples=(
                    WorkedExample(
                        scenario="Label every row",
                        configuration={"labl": "x"},
                        explanation="Every row gains a label column holding 'x'.",
                        inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                        outputs=Ports({"out": pl.DataFrame({"id": [1], "label": ["x"]})}),
                    ),
                )
            ),
            ("does not validate", "labl"),
            ("does not run",),
        ),
        (_StubTool(examples=()), ("no worked examples",), ()),
        (_StubTool(description="  "), ("blank description",), ("field",)),
        (_StubTool(detailed_description=""), ("blank detailed description",), ()),
        (
            _StubTool(template={"label": ConfigField(type="string", description="", required=True)}),
            ("label", "blank description"),
            (),
        ),
        (
            _StubTool(template={"label": ConfigField(type="text", description="The label to add.", required=True)}),
            ("unknown type", "text"),
            (),
        ),
        (_StubTool(field_help={"labl": "The label to add."}), ("field_help", "labl"), ()),
        (_StubTool(output_ports=()), ("no output ports",), ()),
        (_StubTool(input_ports=("in", "in")), ("duplicate input port",), ()),
        (_StubTool(name="Label Rows"), ("machine name",), ()),
        (
            _StubTool(input_ports=("rows",), examples=(conforming_example,)),
            ("does not run", "undeclared input port"),
            ("does not validate",),
        ),
    ]


class TestCatalogConformance:
    """The test that makes the catalog trustworthy: every documented example actually runs."""

    @pytest.mark.parametrize("tool", CONFORMANCE_SUBJECTS, ids=lambda tool: tool.name)
    def test_every_tool_conforms_to_the_catalog_rules(self, tool: Any) -> None:
        """Described, documented, validated, and its worked examples reproduce exactly."""
        check_tool(tool)

    @pytest.mark.parametrize(
        ("tool", "expected_substrings", "forbidden_substrings"),
        _broken_tools(),
        ids=[
            "example_output_has_drifted",
            "example_configuration_does_not_validate",
            "no_worked_examples",
            "blank_description",
            "blank_detailed_description",
            "config_field_without_description",
            "config_field_with_unknown_type",
            "help_for_a_field_that_does_not_exist",
            "no_output_ports",
            "duplicate_port_names",
            "name_is_not_a_machine_name",
            "example_uses_an_undeclared_port",
        ],
    )
    def test_conformance_detects_a_drifted_or_undocumented_tool(
        self,
        tool: Any,
        expected_substrings: tuple[str, ...],
        forbidden_substrings: tuple[str, ...],
    ) -> None:
        """Each rule has teeth: a tool broken in one way is refused, naming the tool and the rule.

        Naming the rule is the half that matters in a catalog of fifteen tools: a refusal that
        says only "this tool is broken" sends a reader to read the whole tool, and a refusal that
        names a rule the tool did not break sends them to the wrong place entirely.
        """
        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message, f"refusal should name the tool: {message}"
        for substring in expected_substrings:
            assert substring in message, f"refusal should name the rule via {substring!r}: {message}"
        for substring in forbidden_substrings:
            assert substring not in message, f"refusal should not mention {substring!r}: {message}"


# ---------------------------------------------------------------------------
# Cycle 2 -- the rules the contract claims but does not yet apply
#
# One theme runs through everything below. A rule that holds only when a tool author remembers to
# apply it is not a rule, it is a convention, and a catalog of fifteen tools inherits the gaps in a
# convention rather than the convention itself. So each rule here is moved to, or pinned at, the one
# place every call passes through -- the boundary and the conformance check -- and the subject used
# to pin it is a tool that does the natural thing rather than the careful thing.
# ---------------------------------------------------------------------------


def _names(message: str, port: str) -> bool:
    """True when ``message`` names ``port`` as a word rather than inside a longer one.

    Needed because the short port names this library uses -- ``in``, ``out`` -- are substrings of
    the words any refusal about them contains. A plain ``"in" in message`` check is satisfied by
    the word "input", so it would pass against a message that never names the port at all, which
    is precisely the defect these tests exist to catch.
    """
    return re.search(rf"\b{re.escape(port)}\b", message) is not None


class _NoRulesTool(_StubTool):
    """A tool with nothing to say beyond its template, so its ``validate_config`` is a no-op.

    This is the shape most tools have: no cross-field rule, therefore no reason to write anything
    in ``validate_config``. It is exactly the shape that reveals whether template validation
    happens because the boundary does it, or only because a tool happened to ask for it. The
    recorder is here so a test can assert *what* the tool was handed, not merely that it was called.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.validated: list[Mapping[str, Any]] = []

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self.validated.append(dict(config))


def _permissive_template() -> dict[str, ConfigField]:
    """A template with one field of each shape that matters: required, defaulted, and neither."""
    return {
        "label": ConfigField(type="string", description="The label to add.", required=True),
        "mode": ConfigField(
            type="string",
            description="How to add it.",
            required=False,
            default="append",
            enum=["append", "replace"],
        ),
        "predicate": ConfigField(type="expression", description="Which rows to label.", required=False),
    }


def _recorder() -> tuple[list[Mapping[str, Any]], Callable[[Ports, Mapping[str, Any]], Ports]]:
    """A list and an execute function that appends the config it was called with to it."""
    seen: list[Mapping[str, Any]] = []

    def execute_fn(inputs: Ports, config: Mapping[str, Any]) -> Ports:
        seen.append(dict(config))
        return Ports({"out": inputs["in"]})

    return seen, execute_fn


def _template_violations() -> list[tuple[Mapping[str, Any], str]]:
    """(config, substring the refusal must contain) for each rule a template can express."""
    return [
        ({"label": "x", "labl": "y"}, "labl"),
        ({"mode": "append"}, "label"),
        ({"label": "${rules.category}"}, "${rules.category}"),
        ({"label": "x", "mode": "sideways"}, "sideways"),
        ({"label": "x", "predicate": "[amount] >"}, "predicate"),
    ]


class TestTheBoundaryEnforcesTheTemplate:
    """Template validation belongs to the boundary, not to each tool's ``validate_config``.

    A tool with no cross-field rules writes ``validate_config`` as a no-op -- the natural thing to
    write when there is nothing to add -- and under the old order that bought it *no* validation at
    all: an unknown field, a missing required field and an unresolved ``${...}`` placeholder all
    reached ``execute``, the last one writing the literal placeholder text into a data column. The
    conformance check cannot catch it either, because it only ever calls ``validate_config`` with a
    configuration that is already valid.
    """

    @pytest.mark.parametrize(
        ("config", "expected_substring"),
        _template_violations(),
        ids=[
            "unknown_field",
            "required_field_missing",
            "unresolved_parameter_placeholder",
            "value_outside_enum",
            "expression_that_does_not_compile",
        ],
    )
    def test_the_boundary_validates_the_template_even_when_the_tool_has_no_rules(
        self,
        config: Mapping[str, Any],
        expected_substring: str,
    ) -> None:
        """Every template-expressible rule is refused by the boundary, and execute never runs."""
        seen, execute_fn = _recorder()
        tool = _NoRulesTool(template=_permissive_template(), execute_fn=execute_fn, examples=())

        with pytest.raises(ConfigError) as caught:
            run_tool(tool, Ports({"in": _frame()}), config)

        assert expected_substring in str(caught.value), f"refusal should name the problem: {caught.value}"
        assert seen == [], "execute must not run on a config the template already refuses"

    def test_the_boundary_order_is_ports_then_template_then_the_tools_own_rules(self) -> None:
        """Ports first, then the template, then the tool's own rules -- on a filled configuration.

        The order is the contract. Ports come first because a port mistake is reported against the
        flow that wired it, not against the configuration. The template comes next, unconditionally.
        The tool's own rules come last and see the *filled* configuration, which is what lets a
        cross-field rule read a defaulted value by subscript instead of guessing with ``.get``.
        """
        seen, execute_fn = _recorder()
        tool = _NoRulesTool(template=_permissive_template(), execute_fn=execute_fn, examples=())

        with pytest.raises(PortError):
            run_tool(tool, Ports({}), {"labl": "y"})
        assert tool.validated == [], "a port problem is reported before the config is looked at"

        result = run_tool(tool, Ports({"in": _frame()}), {"label": "x"})

        assert tool.validated == [{"label": "x", "mode": "append", "predicate": None}], (
            "the tool's own rules must see every template field, defaults filled in"
        )
        assert seen == tool.validated, "execute and validate_config must be handed the same config"
        assert result.names == ("out",)


class TestConformanceComparesEveryDeclaredOutput:
    """A worked example must document every declared output port, and every one is compared.

    The undocumented ports are exactly the side outputs -- the unmatched rows, the leftovers from
    each side -- which are the whole reason the envelope exists and, being anti-joins, the likeliest
    to be wrong. An example that documents one of three outputs is an example that has been *run*
    rather than *checked*, and ``outputs=Ports({})`` is that taken to its limit.
    """

    @staticmethod
    def _two_output_tool(documented: Ports) -> Any:
        """A conforming two-output tool whose example documents ``documented`` as its output."""

        def split(inputs: Ports, config: Mapping[str, Any]) -> Ports:
            frame = inputs["in"]
            return Ports({"out": frame, "spare": frame})

        return _StubTool(
            output_ports=("out", "spare"),
            execute_fn=split,
            examples=(
                WorkedExample(
                    scenario="Send every row to both outputs",
                    configuration={"label": "x"},
                    explanation="Every row leaves on out and on spare, unchanged.",
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=documented,
                ),
            ),
        )

    @pytest.mark.parametrize(
        ("documented", "expected_substrings"),
        [
            pytest.param(("out",), ("spare",), id="a_declared_output_is_undocumented"),
            pytest.param((), ("out", "spare"), id="no_output_is_documented_at_all"),
            pytest.param(("out", "spare", "extra"), ("extra",), id="an_undeclared_output_is_documented"),
        ],
    )
    def test_a_worked_example_must_document_every_declared_output_port(
        self,
        documented: tuple[str, ...],
        expected_substrings: tuple[str, ...],
    ) -> None:
        """An example documenting the wrong set of output ports is a conformance failure by name."""
        frame = pl.DataFrame({"id": [1]})
        tool = self._two_output_tool(Ports({name: frame for name in documented}))

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message
        for port in expected_substrings:
            assert _names(message, port), f"refusal should name the port {port!r}: {message}"


class TestRegistryConformance:
    """The catalog check has to be exercised on a catalog that holds something.

    Run only against the shipped registry -- empty at this stage -- ``check_registry`` passes
    vacuously, so a body of ``pass`` would satisfy the whole suite. A registry holding one
    conforming tool and one broken one is what tells the two apart.
    """

    def test_check_registry_refuses_a_catalog_holding_a_broken_tool(self) -> None:
        """A whole catalog is one call, and the refusal names the tool that broke it."""
        conforming = ToolRegistry()
        conforming.register(_KeepRowsTool())
        conforming.register(_MatchRowsTool())
        check_registry(conforming)

        mixed = ToolRegistry()
        mixed.register(_KeepRowsTool())
        mixed.register(_StubTool(name="odd_one", category="transformation"))

        with pytest.raises(ConformanceError) as caught:
            check_registry(mixed)

        assert "odd_one" in str(caught.value), f"refusal should name the offending tool: {caught.value}"


# ---------------------------------------------------------------------------
# A configuration that nests
#
# The expression seam has to reach an expression wherever it sits, and the tool below is why. Its
# configuration has no top-level expression field at all: every expression it runs is nested inside
# an array of objects. A seam that only looks at top-level fields finds nothing to pre-flight here,
# so a syntactically broken branch survives loading and dies inside execute as an ExpressionError --
# outside the whole error family a caller is told to catch.
# ---------------------------------------------------------------------------


_CASE_WHEN_CONFIG: Mapping[str, Any] = {
    "rules": [
        {"when": "[code] = 'A'", "then": "'counted'"},
        {"when": "[amount] > 100", "then": "'large'"},
    ],
    "otherwise": "'unmatched'",
    "output": "disposition",
}


def _case_when_template() -> dict[str, ConfigField]:
    """The nested template shape: an array whose element is an object of expressions.

    ``items`` describes an array's element and ``fields`` describes an object's members, which is
    the smallest pair of additions that makes this shape -- the common one across ordered rules,
    lists of aggregations, lists of key pairs and lists of output columns -- expressible at all.
    """
    return {
        "rules": ConfigField(
            type="array",
            description="Ordered rules; the first one whose condition holds wins.",
            required=True,
            items=ConfigField(
                type="object",
                description="One rule: a condition, and the value to write where it holds.",
                fields={
                    "when": ConfigField(
                        type="expression",
                        description="Condition over the input columns.",
                        required=True,
                    ),
                    "then": ConfigField(
                        type="expression",
                        description="Value written where the condition holds.",
                        required=True,
                    ),
                },
            ),
        ),
        "otherwise": ConfigField(
            type="expression",
            description="Value written where no rule held.",
            required=True,
        ),
        "output": ConfigField(type="string", description="Name of the column to write.", required=True),
    }


def _case_when_template_with_a_broken_spec(defect: str) -> dict[str, ConfigField]:
    """The nested template above, with one nested *field spec* broken in the named way."""
    template = _case_when_template()
    rules = template["rules"]
    if defect == "nested_blank_description":
        broken = ConfigField(type="expression", description="   ", required=True)
        template["rules"] = ConfigField(
            type=rules.type,
            description=rules.description,
            required=True,
            items=ConfigField(
                type="object",
                description="One rule.",
                fields={"when": broken, "then": rules.items.fields["then"]},
            ),
        )
    elif defect == "nested_unknown_type":
        broken = ConfigField(type="text", description="Condition over the input columns.", required=True)
        template["rules"] = ConfigField(
            type=rules.type,
            description=rules.description,
            required=True,
            items=ConfigField(
                type="object",
                description="One rule.",
                fields={"when": broken, "then": rules.items.fields["then"]},
            ),
        )
    elif defect == "items_on_a_non_array_field":
        template["output"] = ConfigField(
            type="string",
            description="Name of the column to write.",
            required=True,
            items=ConfigField(type="string", description="A member of something that is not an array."),
        )
    elif defect == "fields_on_a_non_object_field":
        template["otherwise"] = ConfigField(
            type="expression",
            description="Value written where no rule held.",
            required=True,
            fields={"when": ConfigField(type="expression", description="A member of something with no members.")},
        )
    else:  # pragma: no cover - a typo in a test id should not pass quietly
        raise AssertionError(f"unknown defect {defect!r}")
    return template


_NESTED_SPEC_EXPECTATIONS: Mapping[str, tuple[str, ...]] = {
    "nested_blank_description": ("rules", "when", "blank description"),
    "nested_unknown_type": ("rules", "when", "unknown type", "text"),
    "items_on_a_non_array_field": ("output", "items"),
    "fields_on_a_non_object_field": ("otherwise", "fields"),
}


class _CaseWhenTool:
    """One input, one output: ordered rules write a label column, and the first match wins."""

    name = "case_when"
    description = "Write a column from ordered rules, taking the first rule that holds."
    category = "transform"
    input_ports = ("rows",)
    output_ports = ("out",)

    def __init__(self, template: Mapping[str, ConfigField] | None = None) -> None:
        self._template = dict(template) if template is not None else None

    def config_template(self) -> Mapping[str, ConfigField]:
        return dict(self._template) if self._template is not None else _case_when_template()

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Walks the rules in order for every row and writes the value of the first rule "
                "whose condition holds, falling back to the otherwise value. Every row keeps its "
                "columns and gains one."
            ),
            examples=(
                WorkedExample(
                    scenario="Label rows by code, then by size",
                    configuration=_CASE_WHEN_CONFIG,
                    explanation=(
                        "The first row matches the first rule. The second row does not, but its "
                        "amount is over the threshold, so the second rule labels it. The third "
                        "matches no rule and takes the otherwise value."
                    ),
                    inputs=Ports({"rows": pl.DataFrame({"code": ["A", "B", "C"], "amount": [1.0, 500.0, 2.0]})}),
                    outputs=Ports(
                        {
                            "out": pl.DataFrame(
                                {
                                    "code": ["A", "B", "C"],
                                    "amount": [1.0, 500.0, 2.0],
                                    "disposition": ["counted", "large", "unmatched"],
                                }
                            )
                        }
                    ),
                ),
            ),
            notes=("Rules are read in order, so a broader rule placed first hides a narrower one after it.",),
            field_help={"rules": "Ordered pairs of a condition and the value to write where it holds."},
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_against_template(self.config_template(), config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Nothing to add beyond the template -- what a tool with no config/schema rule writes."""
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        result = compile_expr(config["otherwise"], frame.columns)
        for rule in reversed(list(config["rules"])):
            condition = compile_expr(rule["when"], frame.columns)
            value = compile_expr(rule["then"], frame.columns)
            result = pl.when(condition).then(value).otherwise(result)
        return Ports({"out": frame.with_columns(result.alias(config["output"]))})


def _nested_config_violations() -> list[tuple[Mapping[str, Any], tuple[str, ...]]]:
    """(config, substrings the refusal must contain) for each rule that must reach inside a value."""
    return [
        (
            {"rules": [{"when": "[amount] >", "then": "'x'"}], "otherwise": "'y'", "output": "d"},
            ("rules[0].when",),
        ),
        (
            {
                "rules": [{"when": "[code] = 'A'", "then": "'x'", "wen": "'z'"}],
                "otherwise": "'y'",
                "output": "d",
            },
            ("rules[0]", "wen"),
        ),
        (
            {"rules": [{"when": "[code] = 'A'"}], "otherwise": "'y'", "output": "d"},
            ("rules[0]", "then"),
        ),
        (
            {"rules": ["[code] = 'A'"], "otherwise": "'y'", "output": "d"},
            ("rules[0]", "object"),
        ),
        (
            {"rules": [{"when": "[code] = 'A'", "then": 5}], "otherwise": "'y'", "output": "d"},
            ("rules[0].then", "expression"),
        ),
    ]


class TestNestedConfigurationShape:
    """A template must describe shape below its top level, because real configurations nest.

    The module already knows this for one rule -- an unresolved ``${...}`` placeholder is hunted
    into a list or an object at any depth -- and not for the others. The cost of the gap is not
    cosmetic: for a configuration built out of ordered rules there is *no* top-level expression
    field, so a loader that pre-flights the fields the template calls expressions pre-flights
    nothing at all, and a broken condition surfaces mid-execute as an error from outside this
    module's error family.
    """

    def test_a_nested_template_declares_and_preflights_an_array_of_objects_of_expressions(self) -> None:
        """The nested shape is declarable, enumerable by path, checkable, and runnable.

        Two questions, two answers. What *may* hold an expression is a property of the template
        alone, which is what a form renderer or a documentation page needs. What *does* hold one,
        and where, needs the configuration too, because only a concrete index -- ``rules[0].when``
        -- is a thing a loader can put in front of a person.
        """
        from dantron_etl.config import expression_values

        tool = _CaseWhenTool()
        template = tool.config_template()

        assert expression_fields(template) == ("rules[].when", "rules[].then", "otherwise")
        assert expression_values(template, _CASE_WHEN_CONFIG) == (
            ("rules[0].when", "[code] = 'A'"),
            ("rules[0].then", "'counted'"),
            ("rules[1].when", "[amount] > 100"),
            ("rules[1].then", "'large'"),
            ("otherwise", "'unmatched'"),
        )

        # What a loader does with the pairs: check every reference against the upstream frame and
        # report the ones that miss by the path they sit at, before a single row is read.
        upstream = pl.DataFrame({"code": ["A"], "amount": [1.0]})
        assert [
            path
            for path, text in expression_values(template, _CASE_WHEN_CONFIG)
            if references(text) - set(upstream.columns)
        ] == []

        typo = {"rules": [{"when": "[amuont] > 0", "then": "'x'"}], "otherwise": "'y'", "output": "d"}
        offending = [
            path for path, text in expression_values(template, typo) if references(text) - set(upstream.columns)
        ]
        assert offending == ["rules[0].when"]

        check_tool(tool)
        result = run_tool(
            tool,
            Ports({"rows": pl.DataFrame({"code": ["A", "B", "C"], "amount": [1.0, 500.0, 2.0]})}),
            _CASE_WHEN_CONFIG,
        )
        assert result["out"]["disposition"].to_list() == ["counted", "large", "unmatched"]

    @pytest.mark.parametrize(
        ("config", "expected_substrings"),
        _nested_config_violations(),
        ids=[
            "nested_expression_does_not_compile",
            "unknown_member_of_a_nested_object",
            "missing_required_member_of_a_nested_object",
            "array_element_of_the_wrong_type",
            "nested_expression_of_the_wrong_type",
        ],
    )
    def test_validate_against_template_checks_a_value_nested_inside_a_config(
        self,
        config: Mapping[str, Any],
        expected_substrings: tuple[str, ...],
    ) -> None:
        """A nested value is checked like a top-level one, and the refusal names its path."""
        with pytest.raises(ConfigError) as caught:
            validate_against_template(_case_when_template(), config)

        message = str(caught.value)
        for substring in expected_substrings:
            assert substring in message, f"refusal should name {substring!r}: {message}"

    @pytest.mark.parametrize(
        "defect",
        list(_NESTED_SPEC_EXPECTATIONS),
        ids=list(_NESTED_SPEC_EXPECTATIONS),
    )
    def test_check_tool_validates_a_nested_field_spec_like_a_top_level_one(self, defect: str) -> None:
        """A nested spec is held to the rules a top-level spec is held to, and fails closed.

        Including the two ways a nested declaration can contradict itself: an element type on a
        field that is not an array, and members on a field that is not an object. Both are silent
        no-ops if unchecked, which means a template that reads as if it constrains something.
        """
        tool = _CaseWhenTool(template=_case_when_template_with_a_broken_spec(defect))

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message
        for substring in _NESTED_SPEC_EXPECTATIONS[defect]:
            assert substring in message, f"refusal should name {substring!r}: {message}"


# ---------------------------------------------------------------------------
# Inputs are read-only
# ---------------------------------------------------------------------------


def _insert_a_column(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Mutate the caller's frame in place, the way ``insert_column`` does, and hand it back."""
    frame = inputs["in"]
    frame.insert_column(frame.width, pl.Series("sneaky", [0] * frame.height))
    return Ports({"out": frame})


def _drop_a_column(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Mutate the caller's frame in place, the way ``drop_in_place`` does, and hand it back."""
    frame = inputs["in"]
    frame.drop_in_place("code")
    return Ports({"out": frame})


class TestInputsAreReadOnly:
    """A tool must not mutate its inputs, and the boundary is where that is enforced.

    ``Ports`` copies the mapping, not the frames, so an in-place column operation reaches straight
    into the caller's own frame. Two things break. In a flow that fans one output into two branches,
    one branch silently rewrites the other branch's input. And because a walkthrough shows the data
    at every step, an in-place change makes an *earlier* step display columns it never produced --
    the reader is shown a lie about what happened, which is worse than an error.

    Comparing each input frame's schema across the call is the cheap enforcement: it catches every
    in-place polars operation that changes the shape, and it costs no copy, which matters when a
    flow is fifteen steps long.
    """

    @pytest.mark.parametrize(
        "execute_fn",
        [
            pytest.param(_insert_a_column, id="insert_column"),
            pytest.param(_drop_a_column, id="drop_in_place"),
        ],
    )
    def test_a_tool_that_mutates_an_input_frame_is_refused_by_name(
        self,
        execute_fn: Callable[[Ports, Mapping[str, Any]], Ports],
    ) -> None:
        """The refusal names the tool and the input port it wrote to."""
        tool = _StubTool(name="mutator", execute_fn=execute_fn, examples=())
        inputs = Ports({"in": pl.DataFrame({"id": [1], "code": ["A"]})})

        with pytest.raises(PortError) as caught:
            run_tool(tool, inputs, {"label": "x"})

        message = str(caught.value)
        assert "mutator" in message, f"refusal should name the tool: {message}"
        assert _names(message, "in"), f"refusal should name the input port: {message}"

    def test_a_tool_that_leaves_its_inputs_alone_is_repeatable(self) -> None:
        """The guard must not fire on a well-behaved tool, and the same Ports can be reused.

        This is the no-false-positive half of the ruling above, and the property that makes a
        fan-out safe: two calls on one ``Ports`` see the same data and produce the same answer.
        """
        frame = pl.DataFrame({"code": ["A", "B"], "amount": [10.0, -5.0]})
        inputs = Ports({"rows": frame})
        tool = _KeepRowsTool()
        config = {"predicate": "[amount] > 0"}

        first = run_tool(tool, inputs, config)
        second = run_tool(tool, inputs, config)

        assert first["matched"].equals(second["matched"])
        assert first["unmatched"].equals(second["unmatched"])
        assert inputs["rows"].columns == ["code", "amount"], "the caller's frame is untouched"
        assert frame.height == 2


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaultsFillTheWholeTemplate:
    """Filling a configuration must be uniform, or ``execute`` reads its own config two ways.

    An optional field with no declared default is absent from a filled configuration, so a tool has
    to reach for it with ``.get`` while subscripting the field next to it -- a distinction with no
    meaning that every tool author has to rediscover field by field. Filling *every* template field,
    with ``None`` where nothing is declared, makes ``execute`` always subscript and makes "was this
    configured?" one expression, ``config["x"] is None``, everywhere.
    """

    def test_apply_defaults_fills_every_field_in_the_template(self) -> None:
        """The filled key set is the template key set, whatever the config left out."""
        template = {
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "mode": ConfigField(type="string", description="How to add it.", required=False, default="append"),
            "note": ConfigField(type="string", description="An optional note.", required=False),
        }

        filled = apply_defaults(template, {"label": "x"})

        assert set(filled) == set(template), "a tool must be able to subscript every template field"
        assert filled == {"label": "x", "mode": "append", "note": None}

        # A supplied value wins over the declared default, including a falsy one.
        assert apply_defaults(template, {"label": "x", "mode": "", "note": "n"}) == {
            "label": "x",
            "mode": "",
            "note": "n",
        }

    def test_a_none_value_is_acceptable_only_where_the_field_is_optional(self) -> None:
        """``None`` means "not configured" for an optional field, and is still refused for a required one.

        This is what makes filling and validating composable: a filled configuration carries
        ``None`` for everything nobody configured, and it has to survive its own type, enum and
        expression checks -- while a required field left empty stays an error.
        """
        template = {
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "mode": ConfigField(type="string", description="How to add it.", enum=["append", "replace"]),
            "predicate": ConfigField(type="expression", description="Which rows to label."),
            "keys": ConfigField(type="array", description="Key columns."),
        }

        validate_against_template(template, {"label": "x", "mode": None, "predicate": None, "keys": None})

        with pytest.raises(ConfigError):
            validate_against_template(template, {"label": None, "mode": None, "predicate": None, "keys": None})

    @pytest.mark.parametrize(
        ("tool", "config"),
        [
            pytest.param(_KeepRowsTool(), {"predicate": "[amount] > 0"}, id="required_field_only"),
            pytest.param(
                _KeepRowsTool(),
                {"predicate": "[amount] > 0", "null_rows": "include"},
                id="every_field_given",
            ),
            pytest.param(_MatchRowsTool(), {"keys": ["key"]}, id="one_of_several_optional_fields"),
            pytest.param(
                _MatchRowsTool(),
                {"left_on": ["a"], "right_on": ["b"]},
                id="two_of_several_optional_fields",
            ),
        ],
    )
    def test_a_filled_config_still_validates_against_its_own_template(
        self,
        tool: Any,
        config: Mapping[str, Any],
    ) -> None:
        """Round trip: whatever validates raw must still validate once the defaults are filled in.

        The property that catches the whole class of mistake, because the boundary now validates,
        then fills, then hands the filled configuration to the tool -- whose own rules validate it
        against the same template a second time.
        """
        template = tool.config_template()

        validate_against_template(template, config)
        validate_against_template(template, apply_defaults(template, config))
        tool.validate_config(apply_defaults(template, config))


class TestDeclaredDefaults:
    """A declared default is the one value in a configuration that never meets the validator.

    Validation runs on what a caller supplied; the default is added afterwards, so it is never type
    checked, never enum checked and never compiled. A default of ``"two"`` on a number field
    therefore reaches ``execute`` as a string it says it will never receive, and a default of
    ``"[amount] >"`` on an expression field reaches it as text that cannot compile. Fifteen tools
    will carry several defaults each, so the check belongs in conformance.
    """

    @pytest.mark.parametrize(
        ("spec", "expected_substrings"),
        [
            pytest.param(
                ConfigField(type="number", description="How many.", default="two"),
                ("size", "default"),
                id="default_of_the_wrong_type",
            ),
            pytest.param(
                ConfigField(type="string", description="Which one.", default="c", enum=["a", "b"]),
                ("size", "default"),
                id="default_outside_the_enum",
            ),
            pytest.param(
                ConfigField(type="expression", description="Which rows.", default="[amount] >"),
                ("size", "default"),
                id="default_that_does_not_compile",
            ),
            pytest.param(
                ConfigField(type="integer", description="How many.", default=3.5),
                ("size", "default"),
                id="fractional_default_for_a_whole_number",
            ),
        ],
    )
    def test_a_declared_default_must_satisfy_its_own_field_spec(
        self,
        spec: ConfigField,
        expected_substrings: tuple[str, ...],
    ) -> None:
        """A default that its own field would reject is refused at conformance, before any run."""
        tool = _StubTool(
            template={
                "label": ConfigField(type="string", description="The label to add.", required=True),
                "size": spec,
            }
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message
        for substring in expected_substrings:
            assert substring in message, f"refusal should name {substring!r}: {message}"


# ---------------------------------------------------------------------------
# The published whitelists
# ---------------------------------------------------------------------------


class TestFieldTypes:
    """The set of field types is published, exact, and fails closed on anything outside it.

    ``validate_against_template`` looks a field's type up and *skips validation when the lookup
    misses* -- a whitelist that fails open, in a package whose expression language is built end to
    end on refusing by name. The cost is silent: a template author writing ``type="integer"``,
    which is the obvious name to reach for, gets a field that accepts anything at all.
    """

    def test_field_types_is_an_exact_set_and_an_unknown_type_fails_closed(self) -> None:
        """The published set is exactly this, and a type outside it is refused in both places."""
        assert FIELD_TYPES == frozenset({"string", "integer", "number", "boolean", "array", "object", "expression"})

        template = {"label": ConfigField(type="text", description="The label to add.", required=True)}

        with pytest.raises(ConfigError) as caught:
            validate_against_template(template, {"label": "x"})
        assert "text" in str(caught.value), f"refusal should name the unknown type: {caught.value}"

        with pytest.raises(ConformanceError):
            check_tool(_StubTool(template=template))

    @pytest.mark.parametrize(
        ("field_type", "value", "accepted"),
        [
            pytest.param("integer", 3, True, id="integer_accepts_a_whole_number"),
            pytest.param("integer", 3.5, False, id="integer_refuses_a_fraction"),
            pytest.param("integer", True, False, id="integer_refuses_a_boolean"),
            pytest.param("integer", "3", False, id="integer_refuses_text"),
            pytest.param("number", 3.5, True, id="number_accepts_a_fraction"),
            pytest.param("number", True, False, id="number_refuses_a_boolean"),
        ],
    )
    def test_a_whole_number_field_accepts_whole_numbers_only(
        self,
        field_type: str,
        value: Any,
        accepted: bool,
    ) -> None:
        """``integer`` is the type a template author reaches for, and a boolean is not one.

        A boolean is an ``int`` in Python, which is exactly the trap the expression language
        already had to sidestep for its own whole-number arguments; the two must agree.
        """
        template = {"size": ConfigField(type=field_type, description="How many.")}

        if accepted:
            validate_against_template(template, {"size": value})
        else:
            with pytest.raises(ConfigError):
                validate_against_template(template, {"size": value})


class TestCategories:
    """A tool's category is checked against a published set, like every other whitelist here.

    Nothing defines the categories today, though two are already in use, and a catalog of fifteen
    tools assembled that way produces ``transform``, ``Transform``, ``transformation`` and ``""`` --
    four groups in a rendered catalog where there is one group in reality.
    """

    def test_categories_is_published_and_membership_is_checked(self) -> None:
        """The set is exactly the catalog's groups, and a tool outside it is refused by name."""
        from dantron_etl.authoring import CATEGORIES

        assert CATEGORIES == frozenset({"input", "transform", "combine", "preparation", "text", "output", "assert"})
        assert _KeepRowsTool().category in CATEGORIES
        assert _MatchRowsTool().category in CATEGORIES

        for category in ("Transform", "transformation", "", "  "):
            with pytest.raises(ConformanceError) as caught:
                check_tool(_StubTool(category=category))
            assert "categor" in str(caught.value), f"refusal should name the rule: {caught.value}"


# ---------------------------------------------------------------------------
# What comes back out of execute
# ---------------------------------------------------------------------------


class TestExecuteReturnValue:
    """A tool that returns something other than ``Ports`` is refused by name at the boundary.

    ``return {"out": frame}`` is the likeliest authoring slip in a library of fifteen tools, and it
    currently surfaces as a bare ``AttributeError`` raised from inside the boundary, naming neither
    the tool nor the mistake -- the same class of defect the envelope exists to prevent, one level up.
    """

    @pytest.mark.parametrize(
        ("returns", "expected_type_name"),
        [
            pytest.param("dict", "dict", id="a_bare_dict_of_frames"),
            pytest.param("frame", "DataFrame", id="a_bare_frame"),
            pytest.param("none", "NoneType", id="nothing_at_all"),
            pytest.param("list", "list", id="a_list_of_frames"),
        ],
    )
    def test_a_tool_that_does_not_return_ports_is_refused_by_name(
        self,
        returns: str,
        expected_type_name: str,
    ) -> None:
        """The refusal names the tool and what it handed back instead."""

        def execute_fn(inputs: Ports, config: Mapping[str, Any]) -> Any:
            frame = inputs["in"]
            return {"dict": {"out": frame}, "frame": frame, "none": None, "list": [frame]}[returns]

        tool = _StubTool(name="wrong_shape", execute_fn=execute_fn, examples=())

        with pytest.raises(PortError) as caught:
            run_tool(tool, Ports({"in": _frame()}), {"label": "x"})

        message = str(caught.value)
        assert "wrong_shape" in message, f"refusal should name the tool: {message}"
        assert expected_type_name in message, f"refusal should name what came back: {message}"


# ---------------------------------------------------------------------------
# A failure inside execute
# ---------------------------------------------------------------------------


def _raises(error: BaseException) -> Callable[[Ports, Mapping[str, Any]], Ports]:
    """An execute that raises ``error`` -- one prepared exception, so the test can identify it."""

    def execute_fn(inputs: Ports, config: Mapping[str, Any]) -> Ports:
        raise error

    return execute_fn


def _a_string_kernel_on_a_number(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """A real call into the query engine, in the shape that found this defect.

    Not a prepared exception: the engine raises this one itself, from a kernel a tool asked for
    against a column that cannot take it. Every tool in this library reaches for the engine like
    this somewhere, which is the reason the translation cannot be left to each of them.
    """
    return Ports({"out": inputs["in"].select(pl.col("id").str.to_uppercase())})


_EXECUTE_MISHAPS: list[Any] = [
    pytest.param(
        _a_string_kernel_on_a_number,
        pl.exceptions.SchemaError,
        "String",
        id="a_real_engine_call_that_the_engine_refuses",
    ),
    pytest.param(
        _raises(pl.exceptions.InvalidOperationError("expected String type, got: cat")),
        pl.exceptions.InvalidOperationError,
        "expected String type",
        id="the_engines_own_exception_type",
    ),
    pytest.param(
        _raises(KeyError("matched")),
        KeyError,
        "matched",
        id="a_key_error_from_a_tools_own_bookkeeping",
    ),
    pytest.param(
        _raises(TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'")),
        TypeError,
        "unsupported operand",
        id="a_type_error_from_a_none_a_tool_did_not_expect",
    ),
    pytest.param(
        _raises(MemoryError("failed to allocate 4 GiB")),
        MemoryError,
        "4 GiB",
        id="a_genuine_resource_failure",
    ),
]


class _BrokenPhaseTool(_StubTool):
    """A conforming tool that breaks in exactly one of the phases the boundary calls.

    One class rather than four stubs, because the point is that the phases are interchangeable as
    far as a caller is concerned: whichever one fails, the caller catches the same family and is
    told which phase it was.
    """

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(name="phase_trouble", execute_fn=_raises(cause))
        self._phase = phase
        self._cause = cause

    def _break_here(self, phase: str) -> None:
        if self._phase == phase:
            raise self._cause

    def config_template(self) -> Mapping[str, ConfigField]:
        self._break_here("config_template")
        return super().config_template()

    def help(self) -> ToolHelp:
        self._break_here("help")
        return super().help()

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self._break_here("validate_config")
        super().validate_config(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        self._break_here("validate_inputs")
        super().validate_inputs(inputs, config)


def _causes(error: BaseException) -> list[BaseException]:
    """Every exception reachable by following ``__cause__``, so a chain of any depth is checkable."""
    chain: list[BaseException] = []
    cause = error.__cause__
    while cause is not None and cause not in chain:
        chain.append(cause)
        cause = cause.__cause__
    return chain


class TestAFailureInsideExecuteJoinsTheErrorFamily:
    """Whatever goes wrong inside ``execute``, the caller catches one family.

    The promise is that a tool's failures arrive as ``ToolError``, so a flow driver has one thing
    to catch. A tool cannot keep that promise on its own: every tool calls into the query engine,
    the engine raises its own exception types, and asking each of fifteen tools to wrap each of its
    own calls is a convention rather than a rule -- fifteen chances to forget, and the first real
    tool forgot. It is the same reasoning that moved template validation onto this boundary and
    then put the expression seam on it: a rule that holds only when an author remembers to apply
    it is not a rule.

    So the boundary translates. Three things make the translation safe rather than merely tidy, and
    all three are pinned below: the original exception survives as ``__cause__``, so no traceback is
    lost; the message carries the failing tool's name *and* what actually went wrong, so a log line
    on its own is still a diagnosis -- an out-of-memory failure must not be reported as though the
    configuration were at fault; and an error a tool raised deliberately passes straight through,
    because burying a tool's own careful refusal under a generic wrapper would cost the author the
    only message written for them.
    """

    @pytest.mark.parametrize(("execute_fn", "cause_type", "expected_text"), _EXECUTE_MISHAPS)
    def test_a_non_tool_error_becomes_an_execution_error_that_names_the_tool_and_the_cause(
        self,
        execute_fn: Callable[[Ports, Mapping[str, Any]], Ports],
        cause_type: type[BaseException],
        expected_text: str,
    ) -> None:
        """One family out, whatever went in -- and nothing about the failure is thrown away.

        ``ExecutionError`` rather than ``ConfigError``: "the engine failed while this tool ran" and
        "this configuration is wrong" have different fixes and different readers, and a caller that
        retries one and reports the other needs to tell them apart. It is still a ``ToolError`` and
        still a ``ValueError``, so a caller that does not care keeps working unchanged.
        """
        tool = _StubTool(name="engine_trouble", execute_fn=execute_fn, examples=())

        with pytest.raises(ExecutionError) as caught:
            run_tool(tool, Ports({"in": _frame()}), {"label": "x"})

        failure = caught.value
        assert isinstance(failure, ToolError), "an execution failure must be catchable as ToolError"
        assert isinstance(failure, ValueError), "an execution failure must be catchable as ValueError"
        assert isinstance(failure.__cause__, cause_type), (
            f"the original failure must survive as __cause__, not be replaced: {failure.__cause__!r}"
        )

        message = str(failure)
        assert "engine_trouble" in message, f"the refusal must name the tool that failed: {message}"
        assert cause_type.__name__ in message, f"the refusal must name what actually failed: {message}"
        assert expected_text in message, f"the original message must be readable in the refusal: {message}"

    @pytest.mark.parametrize(
        "error",
        [
            ConfigError("config field 'predicate' does not give a truth value"),
            PortError("'stub' produced no rows on 'out'"),
            ConformanceError("'stub' has drifted"),
            RegistryError("'stub' is not registered"),
            ToolError("something a tool decided to refuse"),
            ExecutionError("the engine failed and the tool said so itself"),
        ],
        ids=["config", "port", "conformance", "registry", "base", "execution"],
    )
    def test_an_error_a_tool_raises_deliberately_passes_through_unchanged(self, error: ToolError) -> None:
        """A tool's own refusal reaches the caller as the tool wrote it, wrapped in nothing.

        This is the half that makes the translation above safe. A tool that has worked out exactly
        what is wrong with a configuration -- naming the field, quoting the text, saying what was
        needed instead -- has written the most useful sentence anyone will read about that failure.
        Wrapping it in "the engine failed while this tool ran" would replace a precise diagnosis
        with a vague one, and would change its type, so a caller distinguishing a bad configuration
        from a broken run would get the wrong answer for every tool that validates properly.
        """
        tool = _StubTool(name="deliberate", execute_fn=_raises(error), examples=())

        with pytest.raises(ToolError) as caught:
            run_tool(tool, Ports({"in": _frame()}), {"label": "x"})

        assert caught.value is error, "the tool's own error must arrive as itself, not as a copy or a wrapper"
        assert type(caught.value) is type(error), f"the type a tool chose must survive: {type(caught.value)}"
        assert str(caught.value) == str(error), "the message a tool wrote must not be rewritten"
        assert caught.value.__cause__ is None, "an error that passed through was not caused by anything"

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit], ids=["keyboard_interrupt", "system_exit"])
    def test_an_interrupt_is_not_turned_into_a_tool_error(self, interrupt: type[BaseException]) -> None:
        """Asking a long flow to stop must stop it, not report it as a tool that misbehaved.

        A flow of many steps over large frames is the thing somebody interrupts, and a boundary
        that caught everything would swallow that interrupt into an error the driver is written to
        report and continue past. So the translation covers ``Exception`` and not
        ``BaseException``.
        """
        tool = _StubTool(name="interrupted", execute_fn=_raises(interrupt("stop")), examples=())

        with pytest.raises(interrupt):
            run_tool(tool, Ports({"in": _frame()}), {"label": "x"})

    def test_a_worked_example_that_fails_in_the_engine_is_reported_as_a_conformance_error(self) -> None:
        """Catalog conformance keeps working once every failure is in the family it catches.

        ``check_tool`` runs every worked example through ``run_tool`` and catches ``ToolError``, so
        an example whose tool hits the engine used to abort the whole catalog check with the
        engine's own exception -- and a catalog check that dies on the first tool reports nothing
        about the rest. Now it becomes what it is: an example that does not run. The chain is
        pinned as well, because the reader of a failed catalog check needs the engine's own words
        to be still reachable from the report.
        """
        tool = _StubTool(name="engine_trouble_in_an_example", execute_fn=_a_string_kernel_on_a_number)

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert "engine_trouble_in_an_example" in message, f"the report must name the tool: {message}"
        assert "Label every row" in message, f"the report must name the example that failed: {message}"

        execution_failure = caught.value.__cause__
        assert isinstance(execution_failure, ExecutionError), f"the conformance failure's cause: {execution_failure!r}"
        assert isinstance(execution_failure.__cause__, pl.exceptions.PolarsError), (
            f"the engine's own exception must still be reachable: {execution_failure.__cause__!r}"
        )

    def test_a_failure_in_any_phase_of_the_boundary_says_which_phase_it_was(self) -> None:
        """``execute`` is not the only tool code the boundary calls, so it is not the only risk.

        ``run_tool`` asks the tool for its template and then runs the tool's own validation, and
        both are code an author wrote: a ``KeyError`` on a config field they misspelled in a
        cross-field rule is the obvious one, and it happens on this function's call path, before
        ``execute`` is reached. Leaving those two unwrapped would make the promise hold everywhere
        except in the two places a caller cannot see the difference from.

        The phase is named because the three failures send a reader to three different places: a
        template that cannot be built, a validator that broke while checking, and a run that failed
        against real data. The same cause is used for all three here, so the only thing that can
        make the three messages differ is the boundary saying which phase it was.
        """
        cause = KeyError("label")
        failures: dict[str, ExecutionError] = {}

        for phase in ("config_template", "validate_config", "execute"):
            tool = _BrokenPhaseTool(phase, cause)
            with pytest.raises(ExecutionError) as caught:
                run_tool(tool, Ports({"in": _frame()}), {"label": "x"})
            failures[phase] = caught.value

        for phase, failure in failures.items():
            message = str(failure)
            assert isinstance(failure, ToolError), f"the {phase} phase must stay in the family"
            assert failure.__cause__ is cause, f"the {phase} phase lost the original: {failure.__cause__!r}"
            assert "phase_trouble" in message, f"the {phase} refusal must name the tool: {message}"
            assert "KeyError" in message and "label" in message, (
                f"the {phase} refusal must carry what actually failed: {message}"
            )

        lowered = {phase: str(failure).lower() for phase, failure in failures.items()}
        assert "template" in lowered["config_template"], lowered["config_template"]
        assert "valid" in lowered["validate_config"], lowered["validate_config"]
        assert "run" in lowered["execute"], lowered["execute"]
        assert len(set(lowered.values())) == 3, f"the three phases must not read identically: {lowered}"

    def test_a_tool_whose_own_documentation_raises_is_reported_as_a_tool_error(self) -> None:
        """``check_tool`` calls ``help()``, which is also code an author wrote.

        Which subclass this arrives as is a judgement the catalog can make for itself -- a tool
        whose help cannot even be built is arguably non-conformant rather than merely broken -- so
        what is pinned here is what a caller depends on either way: the family, the tool's name,
        and the original failure still reachable. A catalog check that dies with a raw ``KeyError``
        reports nothing about the tools after it, which is the whole cost of leaving this one out.
        """
        cause = KeyError("predicate")
        tool = _BrokenPhaseTool("help", cause)

        with pytest.raises(ToolError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert "phase_trouble" in message, f"the report must name the tool: {message}"
        assert "KeyError" in message and "predicate" in message, f"the report must carry the cause: {message}"
        assert cause in _causes(caught.value), "the original failure must stay reachable from the report"


# ---------------------------------------------------------------------------
# The error family
# ---------------------------------------------------------------------------


class TestErrorHierarchy:
    """The exception hierarchy is the API a caller writes ``except`` against.

    ``ToolError`` rests a compatibility promise on being a ``ValueError`` -- a caller already
    treating configuration problems as ``ValueError`` keeps working without knowing these types
    exist. Nothing asserted it, so reparenting any of them to ``Exception`` would pass the whole
    suite and break every one of those callers.
    """

    @pytest.mark.parametrize(
        "error_type",
        [ToolError, PortError, ConfigError, RegistryError, ConformanceError, ExecutionError],
        ids=["ToolError", "PortError", "ConfigError", "RegistryError", "ConformanceError", "ExecutionError"],
    )
    def test_the_error_family_descends_from_tool_error_and_value_error(
        self,
        error_type: type[Exception],
    ) -> None:
        """Every error this module raises is catchable as ``ToolError`` and as ``ValueError``."""
        assert issubclass(error_type, ToolError)
        assert issubclass(error_type, ValueError)
        assert isinstance(error_type("something is wrong"), ValueError)


# ---------------------------------------------------------------------------
# Worked example text
# ---------------------------------------------------------------------------


class TestWorkedExampleText:
    """A worked example must name its scenario and explain itself.

    The tool's own descriptions are checked for blankness and the example's are not -- yet the
    conformance messages quote the scenario, so a blank one produces "worked example '' has
    drifted", which identifies nothing in a catalog of fifteen tools. And an example with no
    explanation is the thing an example exists to be.
    """

    @pytest.mark.parametrize(
        ("scenario", "explanation", "expected_substring"),
        [
            pytest.param("", "Every row gains a label column holding 'x'.", "scenario", id="blank_scenario"),
            pytest.param("   ", "Every row gains a label column holding 'x'.", "scenario", id="whitespace_scenario"),
            pytest.param("Label every row", "", "explanation", id="blank_explanation"),
            pytest.param("Label every row", "   ", "explanation", id="whitespace_explanation"),
        ],
    )
    def test_a_worked_example_must_name_its_scenario_and_explain_itself(
        self,
        scenario: str,
        explanation: str,
        expected_substring: str,
    ) -> None:
        """A blank scenario or explanation is a conformance failure naming the tool and the rule."""
        tool = _StubTool(
            examples=(
                WorkedExample(
                    scenario=scenario,
                    configuration={"label": "x"},
                    explanation=explanation,
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"out": pl.DataFrame({"id": [1], "label": ["x"]})}),
                ),
            )
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message
        assert expected_substring in message, f"refusal should name the rule: {message}"


# ---------------------------------------------------------------------------
# The expression seam, pinned so that a shortcut cannot satisfy it
# ---------------------------------------------------------------------------


class TestExpressionFieldSeamHasTeeth:
    """The seam has to find the expression fields, not the first field.

    Pinned against a template whose expression fields are the second and third of three, so that
    neither "the first field" nor "every field" satisfies it. It is also the guard on the nested
    work above: whoever teaches this function to walk into an array must leave a flat template
    reporting exactly the names it reports today.
    """

    def test_expression_fields_finds_every_expression_and_not_merely_the_first_field(self) -> None:
        """Two expression fields out of three, neither of them first, in template order."""
        template = {
            "output": ConfigField(type="string", description="Name of the column to write.", required=True),
            "predicate": ConfigField(type="expression", description="Which rows to write.", required=True),
            "value": ConfigField(type="expression", description="What to write.", required=True),
        }

        assert expression_fields(template) == ("predicate", "value")
        assert expression_fields({}) == ()


# ---------------------------------------------------------------------------
# Cycle 3 -- the two things the catalog cannot yet say
#
# Both are the same shape of gap: a property a tool documents in prose, cannot declare, and
# therefore cannot have checked anywhere a reader looks. Row order is one -- three of the tools
# shipped or in flight promise to preserve it and the conformance check compares with
# ``check_row_order=False``, so the promise is untested exactly where it is made. Non-emptiness is
# the other -- ``required=True`` is satisfied by ``[]``, so a field that needs content has to
# refuse emptiness in its own ``validate_config``, which is a convention rather than a rule.
# ---------------------------------------------------------------------------


def _label_rows_reversed(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """The stub's work, with the rows handed back in the wrong order: right rows, wrong sequence.

    The only defect an order-preserving tool can have that ``check_row_order=False`` cannot see,
    and the one that broke twelve of one tool's own cases while its catalog entry stayed green.
    """
    return Ports({"out": inputs["in"].with_columns(pl.lit(config["label"]).alias("label")).reverse()})


def _label_rows_reversed_when_asked(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Reverse the rows only for the configuration that asks for it, leaving the other alone.

    What makes "the flag is per example" a testable claim: one tool, two examples, and only one of
    them running through the defect.
    """
    work = _label_rows_reversed if config["label"] == "flip" else _label_rows
    return work(inputs, config)


def _three_rows_example(label: str, *, ordered: bool) -> WorkedExample:
    """An example whose documented output has three distinguishable rows, so order is visible."""
    return WorkedExample(
        scenario=f"Label three rows, {label}",
        configuration={"label": label},
        explanation="Every row gains a label column, and the rows leave in the order they arrived.",
        inputs=Ports({"in": pl.DataFrame({"id": [1, 2, 3]})}),
        outputs=Ports({"out": pl.DataFrame({"id": [1, 2, 3], "label": [label] * 3})}),
        check_row_order=ordered,
    )


def _split_in_two(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Two ports out of one input: the first two rows, and the last two, both in arrival order."""
    frame = inputs["in"]
    return Ports({"kept": frame.head(2), "rest": frame.tail(2)})


def _split_reversing(port: str) -> Callable[[Ports, Mapping[str, Any]], Ports]:
    """:func:`_split_in_two` with exactly one of its two ports handed back in the wrong order."""

    def execute_fn(inputs: Ports, config: Mapping[str, Any]) -> Ports:
        produced = _split_in_two(inputs, config)
        return Ports({name: frame.reverse() if name == port else frame for name, frame in produced.items()})

    return execute_fn


def _split_dropping_a_port(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """One of the two declared ports, so the port set is wrong rather than the row order."""
    return Ports({"kept": inputs["in"].head(2)})


def _two_port_example(*, ordered: bool) -> WorkedExample:
    """An ordered example over two ports, each with two rows a reversal would visibly disturb."""
    return WorkedExample(
        scenario="Split four rows into the first two and the last two",
        configuration={"label": "x"},
        explanation="Both outputs keep the order the rows arrived in, which is what makes the split readable.",
        inputs=Ports({"in": pl.DataFrame({"id": [1, 2, 3, 4]})}),
        outputs=Ports(
            {
                "kept": pl.DataFrame({"id": [1, 2]}),
                "rest": pl.DataFrame({"id": [3, 4]}),
            }
        ),
        check_row_order=ordered,
    )


def _two_port_tool(execute_fn: Callable[[Ports, Mapping[str, Any]], Ports], *, ordered: bool) -> Any:
    """A stub declaring two output ports, documented by a single example over both of them."""
    return _StubTool(
        output_ports=("kept", "rest"), examples=(_two_port_example(ordered=ordered),), execute_fn=execute_fn
    )


def _other_drifts() -> list[Any]:
    """Every difference that is *not* row order, applied to an example that asks for row order.

    Each of these is already caught today. They are pinned again under the flag because the
    cheapest wrong implementation of it is to compare something weaker than the default did --
    sorted values, or values without dtypes -- which would trade one blind spot for four.
    """
    return [
        pytest.param(lambda frame: frame.with_columns(pl.col("id") * 10), id="a_changed_value"),
        pytest.param(lambda frame: frame.with_columns(pl.col("id").cast(pl.Float64)), id="a_changed_dtype"),
        pytest.param(lambda frame: frame.select("label", "id"), id="a_swapped_column_order"),
        pytest.param(lambda frame: frame.head(2), id="a_dropped_row"),
        pytest.param(lambda frame: pl.concat([frame, frame.head(1)]), id="a_repeated_row"),
    ]


class TestAnExampleCanDeclareItsRowOrderSignificant:
    """A tool that preserves row order must be able to demonstrate it where a reader looks.

    ``_check_example`` compares with ``check_row_order=False``, which is the right default -- for a
    tool that pairs two sets of rows, order is an artefact of the engine and pinning it would make
    every such example brittle. But it leaves an order-preserving tool unable to have its one
    interesting property checked in the catalog at all: the rows are right, the counts are right,
    the dtypes are right, and the sequence a reader is copying the example for is unchecked.

    So the declaration lives on the *example*, defaults to off, and is named after the polars
    argument it controls so nobody has to learn a second word for it.
    """

    def test_row_order_is_checked_per_example_and_only_where_the_example_asks(self) -> None:
        """One tool, two examples, one defect -- and which example declared it decides the outcome.

        The three pins that matter are all here, on the same pair of tools, because they are one
        claim: an example that asks for row order and gets it passes; an example that does not ask
        keeps today's behaviour even when the rows come back reversed; and neither example's
        declaration reaches the other, so turning it on for one tool's ordered example cannot make
        every join example in the catalog brittle.
        """
        passes = _StubTool(
            examples=(_three_rows_example("keep", ordered=True), _three_rows_example("flip", ordered=False)),
            execute_fn=_label_rows_reversed_when_asked,
        )

        check_tool(passes)

        refuses = _StubTool(
            examples=(_three_rows_example("keep", ordered=False), _three_rows_example("flip", ordered=True)),
            execute_fn=_label_rows_reversed_when_asked,
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(refuses)

        message = str(caught.value)
        assert refuses.name in message, f"refusal should name the tool: {message}"
        assert "Label three rows, flip" in message, f"refusal should name the failing scenario: {message}"
        assert "Label three rows, keep" not in message, (
            f"the example that did not ask for row order is not the one that failed: {message}"
        )
        assert _names(message, "out"), f"refusal should name the port that drifted: {message}"

    def test_an_ordered_example_covers_every_port_it_documents(self) -> None:
        """The flag is per example, and an example documents a tool's whole output, not one port.

        A tool with an ordered port and an unordered one is a real shape, and this is the grain
        that serves it: such a tool documents the ordered claim in an example whose *every* port is
        ordered, and demonstrates the unordered port in an example that does not ask. Per-port
        would put the answer in the tool rather than in the demonstration, where a reader who
        copies an example would still have nothing checked for them.
        """
        check_tool(_two_port_tool(_split_in_two, ordered=True))

        for port in ("kept", "rest"):
            with pytest.raises(ConformanceError) as caught:
                check_tool(_two_port_tool(_split_reversing(port), ordered=True))
            message = str(caught.value)
            assert _names(message, port), f"refusal should name the port whose order was wrong: {message}"

        with pytest.raises(ConformanceError) as caught:
            check_tool(_two_port_tool(_split_dropping_a_port, ordered=True))
        assert _names(str(caught.value), "rest"), (
            f"asking for row order must not skip the port checks that run before the comparison: {caught.value}"
        )

    @pytest.mark.parametrize("drift", _other_drifts())
    def test_asking_for_row_order_still_checks_everything_the_default_checked(
        self,
        drift: Callable[[pl.DataFrame], pl.DataFrame],
    ) -> None:
        """Values, dtypes, column order and row count are all still compared under the flag."""
        tool = _StubTool(
            examples=(_three_rows_example("x", ordered=True),),
            execute_fn=lambda inputs, config: Ports({"out": drift(_label_rows(inputs, config)["out"])}),
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        assert "Label three rows, x" in str(caught.value), f"refusal should name the scenario: {caught.value}"


# ---------------------------------------------------------------------------
# A field that needs content can say so
# ---------------------------------------------------------------------------


def _non_empty_cases() -> list[Any]:
    """(template, config, the path a refusal must name or ``None`` for "this is accepted")."""
    optional_list = {
        "columns": ConfigField(type="array", description="Columns to select.", non_empty=True),
    }
    return [
        pytest.param(
            {"columns": ConfigField(type="array", description="Columns to select.", required=True)},
            {"columns": []},
            None,
            id="required_alone_accepts_an_empty_array",
        ),
        pytest.param(
            {"columns": ConfigField(type="array", description="Columns to select.", required=True, non_empty=True)},
            {"columns": []},
            "columns",
            id="non_empty_refuses_an_empty_array",
        ),
        pytest.param(
            {"label": ConfigField(type="string", description="The label to add.", required=True, non_empty=True)},
            {"label": ""},
            "label",
            id="non_empty_refuses_an_empty_string",
        ),
        pytest.param(
            {"members": ConfigField(type="object", description="Named members.", required=True, non_empty=True)},
            {"members": {}},
            "members",
            id="non_empty_refuses_an_empty_object",
        ),
        pytest.param(optional_list, {"columns": ["code"]}, None, id="non_empty_accepts_a_value_with_content"),
        pytest.param(optional_list, {}, None, id="non_empty_is_silent_about_an_absent_optional_field"),
        pytest.param(optional_list, {"columns": None}, None, id="non_empty_survives_a_filled_none"),
        pytest.param(
            {
                "rules": ConfigField(
                    type="array",
                    description="Ordered rules.",
                    items=ConfigField(
                        type="object",
                        description="One rule.",
                        fields={"on": ConfigField(type="array", description="Key columns.", non_empty=True)},
                    ),
                )
            },
            {"rules": [{"on": []}]},
            "rules[0].on",
            id="non_empty_reaches_a_nested_field_and_names_its_path",
        ),
    ]


class TestAFieldCanDeclareThatItNeedsContent:
    """``required=True`` means "a value arrived", never "the value carries anything".

    That is the correct reading -- ``[]`` is a value, and only the field's own author knows whether
    an empty one is meaningful -- but every array field in this library has to answer the question,
    the first one to need an answer hand-rolled it in its own ``validate_config``, and the contract
    said nothing either way. Non-emptiness is a statement about a value's shape, needing neither
    another field nor the frame, which is exactly the line this module already draws between what a
    template expresses and what a tool's own rules do. So it moves to the template, where it is
    declared once and enforced for everybody, and ``required`` keeps meaning what it means.
    """

    @pytest.mark.parametrize(("template", "config", "named"), _non_empty_cases())
    def test_non_empty_refuses_a_present_but_empty_value_and_required_alone_does_not(
        self,
        template: Mapping[str, ConfigField],
        config: Mapping[str, Any],
        named: str | None,
    ) -> None:
        """Two orthogonal questions: was a value given, and does the value carry anything."""
        if named is None:
            validate_against_template(template, config)
            return

        with pytest.raises(ConfigError) as caught:
            validate_against_template(template, config)

        assert named in str(caught.value), f"refusal should name the field by its path: {caught.value}"

    @pytest.mark.parametrize(
        ("template", "expected_substrings"),
        [
            pytest.param(
                {
                    "label": ConfigField(type="string", description="The label to add.", required=True),
                    "columns": ConfigField(type="array", description="Columns to select.", non_empty=True, default=[]),
                },
                ("columns", "default"),
                id="a_default_its_own_declaration_refuses",
            ),
            pytest.param(
                {
                    "label": ConfigField(type="string", description="The label to add.", required=True),
                    "count": ConfigField(type="integer", description="How many.", non_empty=True),
                },
                ("count", "integer"),
                id="a_declaration_on_a_type_with_no_notion_of_emptiness",
            ),
        ],
    )
    def test_a_non_empty_declaration_that_constrains_nothing_is_a_conformance_failure(
        self,
        template: Mapping[str, ConfigField],
        expected_substrings: tuple[str, ...],
    ) -> None:
        """Declared where it cannot bite, it is a template that reads as if it constrains something.

        The same rule the module already applies to an element type outside an array and to members
        outside an object, for the same reason: the silent no-op is the worst outcome available. An
        empty default on a field that refuses emptiness is the other half -- a default is the one
        value in a configuration that never meets the validator, so it is checked here or nowhere.
        """
        check_tool(
            _StubTool(
                template={
                    "label": ConfigField(type="string", description="The label to add.", required=True, non_empty=True)
                }
            )
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(_StubTool(template=template))

        message = str(caught.value)
        for substring in expected_substrings:
            assert substring in message, f"refusal should say {substring!r}: {message}"


# ---------------------------------------------------------------------------
# A schema probe, and a predicate compiler built on it
#
# Four tools ask the same question -- what does this expression produce, before a row is read --
# and the fourth spelling of it is where they start to disagree. Filter runs
# ``frame.lazy().select(expr).collect_schema()`` and reads the first entry; an ordered-rules tool
# needs it per condition and again to check its branches agree; a date assembler needs it once per
# date part; an assertion tool needs it to refuse a non-condition. So the probe moves next to
# ``compile_for``, and the predicate check is built on top of it rather than beside it.
# ---------------------------------------------------------------------------


def _typed_frame() -> pl.DataFrame:
    """One frame carrying a whole number, text, a number and a truth value, each with a null."""
    return pl.DataFrame(
        {
            "key": [3, 1, 2],
            "code": ["A", "B", None],
            "amount": [10.0, -5.0, 0.0],
            "posted": [True, False, None],
        }
    )


def _dtype_cases() -> list[Any]:
    """(expression, the column it is named after, the dtype it produces)."""
    return [
        pytest.param("[amount] > 0", "amount", pl.Boolean, id="a_comparison_is_a_truth_value"),
        pytest.param("[posted]", "posted", pl.Boolean, id="a_bare_truth_column_is_one_already"),
        pytest.param("upper([code])", "code", pl.String, id="a_text_function_over_text"),
        pytest.param("[amount] * 2", "amount", pl.Float64, id="arithmetic_keeps_the_number"),
        pytest.param("[key]", "key", pl.Int64, id="a_bare_whole_number_column"),
        pytest.param("'x'", "literal", pl.String, id="a_constant_is_named_after_no_column"),
    ]


def _predicate_cases() -> list[Any]:
    """(expression, the column a refusal must name, or ``None`` where the expression is accepted)."""
    return [
        pytest.param("[amount] > 0", None, id="a_comparison_is_accepted"),
        pytest.param("[posted]", None, id="a_bare_truth_column_is_accepted"),
        pytest.param("[key] > 0 and [amount] > 0", None, id="two_conditions_are_accepted"),
        pytest.param("[amount]", "amount", id="a_number_is_not_a_truth_value"),
        pytest.param("upper([code])", "code", id="text_is_not_a_truth_value"),
        pytest.param("'x'", "literal", id="a_constant_string_is_not_a_truth_value"),
    ]


class TestTheSchemaProbeAndThePredicateCompiler:
    """What an expression produces is a schema question, asked once, in one place.

    ``compiled_dtype`` hands back ``(column_name, dtype)`` rather than the dtype alone, because
    every caller that refuses a dtype has to name the column in the refusal -- Filter's own message
    quotes it today -- and a caller wanting only the dtype pays one index for it. It hands back a
    pair rather than a whole schema because a compiled expression produces exactly one column, and
    a mapping would make every caller re-derive "the first and only entry", which is the line of
    code being removed.

    Both helpers are pinned against a frame with **no rows at all**. That axis is what proves a
    refusal was read off the schema rather than off the data: a probe that collected rows would
    still answer on an empty frame, but a check that depended on a value would not.
    """

    @pytest.mark.parametrize(("text", "column", "dtype"), _dtype_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_compiled_dtype_reports_the_column_and_the_dtype_without_reading_a_row(
        self,
        text: str,
        column: str,
        dtype: pl.DataType,
        height: int,
    ) -> None:
        """The pair is the contract: the name for the message, the dtype for the decision."""
        frame = _typed_frame().head(height)

        assert compiled_dtype(frame, text, "value") == (column, dtype)

    @pytest.mark.parametrize(("text", "named"), _predicate_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_compile_predicate_for_refuses_anything_that_is_not_a_truth_value(
        self,
        text: str,
        named: str | None,
        height: int,
    ) -> None:
        """A predicate that is not a condition is the slip every filtering tool has to refuse.

        ``[amount]`` is what an author writes when they meant ``[amount] > 0``: it compiles, because
        that is how a column reference is spelled, and handed to the query engine as a mask it
        raises about a dtype from a library the author has never heard of. The refusal has to name
        the column, the config field to open, and the text to fix -- all three, because this is read
        later in a log by somebody holding a file of many expressions.
        """
        frame = _typed_frame().head(height)

        if named is None:
            expression = compile_predicate_for(frame, text, "predicate")
            assert isinstance(expression, pl.Expr)
            assert frame.filter(expression.fill_null(False)).height <= frame.height, "it is usable as a mask"
            return

        with pytest.raises(ConfigError) as caught:
            compile_predicate_for(frame, text, "predicate")

        message = str(caught.value)
        assert named in message, f"the refusal must name the column: {message}"
        assert "predicate" in message, f"the refusal must name the config field: {message}"
        assert text in message, f"the refusal must quote the expression: {message}"
        _, dtype = compiled_dtype(frame, text, "predicate")
        assert str(dtype) in message, f"the refusal must say what the expression produced instead: {message}"

    @pytest.mark.parametrize(
        "helper",
        [compiled_dtype, compile_predicate_for],
        ids=["compiled_dtype", "compile_predicate_for"],
    )
    @pytest.mark.parametrize(
        "text",
        ["upper([amount])", "[nope] > 0", "[amount] >"],
        ids=["a_text_function_on_a_number", "a_column_that_is_not_there", "text_that_does_not_parse"],
    )
    def test_both_helpers_refuse_in_this_modules_family_and_never_the_engines(
        self,
        helper: Callable[[pl.DataFrame, str, str], Any],
        text: str,
    ) -> None:
        """A caller catches ``ToolError``; an ``ExpressionError`` or a ``PolarsError`` slips past it.

        ``check_tool``'s own example runner is exactly such a caller, so a helper that let either
        escape would take a whole catalog check down with an exception type it does not catch.
        """
        assert not issubclass(ExpressionError, ToolError), "which is why the translation has to happen"

        with pytest.raises(ConfigError) as caught:
            helper(_typed_frame(), text, "value")

        assert isinstance(caught.value, ToolError)
        assert not isinstance(caught.value, (ExpressionError, pl.exceptions.PolarsError))
        assert "value" in str(caught.value), f"the refusal must name the config field: {caught.value}"


# ---------------------------------------------------------------------------
# An error for "the data is impossible"
#
# The family says the configuration is wrong, the wiring is wrong, the catalog is wrong, or
# something unexpected broke. None of them says *the data* was wrong -- and two tools need exactly
# that: an assertion exists to fail a run when an invariant does not hold, and a row-level
# ``on_invalid: error`` policy is a deliberate refusal about values, not about configuration.
# ---------------------------------------------------------------------------


def _offending(rows: int) -> pl.DataFrame:
    """A frame of ``rows`` offending rows, each identifiable by its own key."""
    return pl.DataFrame({"key": [f"k{index}" for index in range(rows)], "part": [None] * rows})


class TestADataErrorSaysTheDataWasWrong:
    """A fifth member, because a driver's response to it differs from its response to the rest.

    A data fault means look at the data; an execution fault means look at the run. Today both
    arrive as ``ExecutionError``, so a driver that continues past a broken run would also continue
    past a failed invariant, and one that stops on a failed invariant would stop on a machine that
    ran out of memory. ``DataError`` is the name chosen over ``AssertionFailure`` (too narrow -- a
    date part that cannot be assembled is no assertion) and over ``ValidationError`` (already the
    word this module uses for configuration validation).
    """

    def test_a_data_error_is_in_the_family_and_apart_from_an_execution_error(self) -> None:
        """Catchable as ``ToolError`` and as ``ValueError``, and never mistaken for the other one."""
        assert issubclass(DataError, ToolError)
        assert issubclass(DataError, ValueError)
        assert not issubclass(DataError, ExecutionError), "a data fault is not an unexpected failure"
        assert not issubclass(ExecutionError, DataError), "and an unexpected failure is not a data fault"

        error = DataError("a row could not be dated", offending=_offending(1), total=1)
        assert isinstance(error, ValueError)
        assert not isinstance(error, ExecutionError)

    def test_run_tool_hands_a_data_error_back_unwrapped_but_still_wraps_an_engine_failure(self) -> None:
        """A deliberate refusal about data is the message that matters, so nothing may bury it.

        Wrapped in ``ExecutionError`` it would read "the tool failed while running", which sends the
        reader to the run when the answer is in the data, and it would change its type, so the one
        distinction a driver needs would be gone.
        """
        refusal = DataError("3 rows hold a month outside 1-12", offending=_offending(3), total=5)
        deliberate = _StubTool(name="refuser", execute_fn=_raises(refusal), examples=())

        with pytest.raises(DataError) as caught:
            run_tool(deliberate, Ports({"in": _frame()}), {"label": "x"})

        assert caught.value is refusal, "a deliberate refusal must arrive as itself"
        assert str(caught.value) == str(refusal), "its message must not be rewritten"
        assert caught.value.__cause__ is None, "it was not caused by anything -- it is the diagnosis"

        broken = _StubTool(name="engine_trouble", execute_fn=_a_string_kernel_on_a_number, examples=())
        with pytest.raises(ExecutionError) as unexpected:
            run_tool(broken, Ports({"in": _frame()}), {"label": "x"})
        assert not isinstance(unexpected.value, DataError), "an engine failure is not a data fault"

    def test_a_data_error_reports_how_many_rows_offended_out_of_how_many_and_caps_the_example(self) -> None:
        """What makes it actionable in a log read by somebody who was not there.

        Three numbers and no more: how many rows offended, out of how many, and an example small
        enough to print. "an invariant failed" sends a reader to a query they have to write; "3 of 5
        rows, first of them k0" sends them to the rows. The cap is the other half -- an assertion
        failing on every row of a large frame must not print the frame into the log.
        """
        error = DataError("a month outside 1-12", offending=_offending(3), total=5)

        assert error.offending_rows == 3
        assert error.total_rows == 5
        assert "3" in str(error) and "5" in str(error), f"both counts must be readable: {error}"
        assert "k0" in str(error), f"an example row must be there to find them by: {error}"

        every_row = 40_000
        capped = DataError("a month outside 1-12", offending=_offending(every_row), total=every_row)

        assert 1 <= MAX_REPORTED_ROWS <= 20, "a cap nobody can read is not a cap"
        assert capped.offending_rows == every_row and capped.total_rows == every_row
        assert len(capped.examples) == MAX_REPORTED_ROWS, "the example is capped, the count is not"
        assert str(every_row) in str(capped), f"the count survives the cap: {capped}"
        assert "k39999" not in str(capped), "the last offending row must not be printed"
        assert len(str(capped)) < 1_000, f"a message of {len(str(capped))} characters is a log nobody reads"


# ---------------------------------------------------------------------------
# An enum value YAML would rewrite is a conformance failure
#
# Pipeline definitions are authored in YAML, which resolves certain bare words into other types
# before this library sees them. Three tools have hand-rolled the check that their own enums avoid
# those words -- and the null spellings were missed by all of them, which is the whole argument for
# moving it: ``on_invalid: null`` arrives as ``None``, ``apply_defaults`` treats it as unset, and
# the policy an author asked for silently becomes the default.
# ---------------------------------------------------------------------------


def _enum_value_cases() -> list[Any]:
    """(enum value, whether the catalog must refuse it) -- every spelling YAML would rewrite."""
    booleans = ("true", "True", "TRUE", "false", "False", "yes", "Yes", "YES", "no", "No", "on", "On", "off", "OFF")
    single_letters = ("y", "Y", "n", "N")
    nulls = ("null", "Null", "NULL", "~", "")
    numbers = ("1", "0", "1.5", "-2")
    safe = ("exclude", "include", "first", "last", "clamp", "set_null", "error", "no_action", "offset", "year")
    return [
        *[pytest.param(value, True, id=f"yaml_boolean_{value or 'blank'}") for value in booleans],
        *[pytest.param(value, True, id=f"yaml_boolean_letter_{value}") for value in single_letters],
        *[pytest.param(value, True, id=f"yaml_null_{value.strip('~') or 'tilde_or_blank'}") for value in nulls],
        *[pytest.param(value, True, id=f"yaml_number_{value}") for value in numbers],
        *[pytest.param(value, False, id=f"safe_{value}") for value in safe],
    ]


def _enum_tool(value: str) -> Any:
    """A conforming tool whose only enum offers ``value`` as one of its two members."""
    return _StubTool(
        template={
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "mode": ConfigField(type="string", description="How to add it.", enum=("normal", value)),
        }
    )


def _labelled_on_a_reserved_port(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """The stub's ordinary work, on an output port named with a word YAML would rewrite."""
    return Ports({"no": inputs["in"].with_columns(pl.lit(config["label"]).alias("label"))})


class TestAnEnumValueYamlWouldRewrite:
    """An author reaches for a policy by its obvious name, and the file hands over another type.

    ``nulls: false`` becomes a boolean and is refused with a type error that has nothing to do with
    what the author meant, which is merely confusing. ``on_invalid: null`` becomes ``None``, is
    treated as unset, and the run proceeds under the default policy -- which is the dangerous one,
    because nothing is refused at all. So the catalog refuses the value at authoring time, where
    there is a person to tell.
    """

    @pytest.mark.parametrize(("value", "refused"), _enum_value_cases())
    def test_check_tool_refuses_an_enum_value_yaml_would_reinterpret(self, value: str, refused: bool) -> None:
        """The safe values are half the test: a rule satisfied by refusing everything is no rule."""
        if not refused:
            check_tool(_enum_tool(value))
            return

        with pytest.raises(ConformanceError) as caught:
            check_tool(_enum_tool(value))

        message = str(caught.value)
        assert "mode" in message, f"the refusal must name the field whose enum it is: {message}"
        assert "YAML" in message or "yaml" in message, f"the refusal must say why: {message}"

    def test_an_output_port_name_yaml_would_rewrite_stays_legal(self) -> None:
        """An output port name is read as a value, where a definition file leaves it alone.

        The rule is about which side of the colon a name appears on. A step names an output port on
        the right, as a value, and a file hands the word over as written. An input port name sits
        on the left, as a key, and is refused -- see the input port rule below. ``filter`` ships
        output ports called ``true`` and ``false``.
        """
        assert {"true", "false", "yes", "no", "on", "off", "null", "~", ""} <= YAML_UNSAFE_VALUES

        check_tool(
            _StubTool(
                output_ports=("no",),
                execute_fn=_labelled_on_a_reserved_port,
                examples=(
                    WorkedExample(
                        scenario="Label every row on a port named for a word YAML rewrites",
                        configuration={"label": "x"},
                        explanation="The port name is wiring, not a value in a configuration file.",
                        inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                        outputs=Ports({"no": pl.DataFrame({"id": [1], "label": ["x"]})}),
                    ),
                ),
            )
        )


# ---------------------------------------------------------------------------
# A config field name YAML would rewrite is a conformance failure
#
# The rule above refuses an enum VALUE a definition file would hand over as another type. A field
# NAME has the same problem and nothing checks it. ``cross_tab`` declares a field called ``on``, so
# a step written as ``on: period`` hands the loader the boolean ``True`` as its key. The name has
# stopped being a name. Quoting it in the file works, but that puts the fix on every author of
# every pipeline that uses the tool. A tool should not be able to declare such a name.
# ---------------------------------------------------------------------------


def _unsafe_field_name_cases() -> list[Any]:
    """Every published unsafe word, plus the bare numbers a file also hands over as another type."""
    return [
        *[
            pytest.param(name, id=f"unsafe_{name.strip('~') or 'tilde_or_blank'}")
            for name in sorted(YAML_UNSAFE_VALUES)
        ],
        *[pytest.param(name, id=f"number_{name}") for name in ("1", "0", "1.5", "-2")],
    ]


def _tool_with_field_named(name: str) -> Any:
    """A tool conforming in every other way, declaring one field called ``name``."""
    return _StubTool(
        template={
            "label": ConfigField(type="string", description="The label to add.", required=True),
            name: ConfigField(type="string", description="A field whose name may not survive a file."),
        }
    )


def _tool_with_member_named(name: str) -> Any:
    """The same field name, one level down, inside an object."""
    return _StubTool(
        template={
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "options": ConfigField(
                type="object",
                description="Options for the step.",
                fields={name: ConfigField(type="string", description="A member whose name may not survive a file.")},
            ),
        }
    )


def _tool_with_element_member_named(name: str) -> Any:
    """The same field name again, inside an array of objects -- where a template is least read."""
    return _StubTool(
        template={
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "rules": ConfigField(
                type="array",
                description="Rules applied in order.",
                items=ConfigField(
                    type="object",
                    description="One rule.",
                    fields={
                        name: ConfigField(type="string", description="A member whose name may not survive a file.")
                    },
                ),
            ),
        }
    )


class TestAConfigFieldNameYamlWouldRewrite:
    """A tool may not declare a field name a definition file turns into something else.

    This was hit writing a real pipeline step. ``cross_tab`` has a field called ``on``, and the
    obvious way to write that step gives the loader the key ``True``. Every author of every
    pipeline using the tool has to know to quote it, and nothing tells them. So the refusal belongs
    on the tool, where there is one author to tell instead of many.
    """

    @pytest.mark.parametrize("name", _unsafe_field_name_cases())
    def test_check_tool_refuses_an_unsafe_top_level_field_name(self, name: str) -> None:
        """A top-level field named for a word a file rewrites is refused at authoring time."""
        with pytest.raises(ConformanceError) as caught:
            check_tool(_tool_with_field_named(name))

        message = str(caught.value)
        assert "YAML" in message or "yaml" in message, f"the refusal must say why: {message}"
        if name.strip():
            assert name in message, f"the refusal must name the field it is about: {message}"

    @pytest.mark.parametrize("name", ["on", "no", "null"])
    def test_check_tool_refuses_an_unsafe_field_name_inside_an_object(self, name: str) -> None:
        """A member of an object is a key in the file too, so the same rule has to reach it.

        A few names rather than all of them: which words are unsafe is settled above, and this is
        about where the rule looks.
        """
        with pytest.raises(ConformanceError) as caught:
            check_tool(_tool_with_member_named(name))

        message = str(caught.value)
        assert "options" in message and name in message, f"the refusal must path the member: {message}"

    @pytest.mark.parametrize("name", ["on", "no", "null"])
    def test_check_tool_refuses_an_unsafe_field_name_inside_an_array_of_objects(self, name: str) -> None:
        """An ordered list of small objects is the common shape, and its keys come from the file.

        ``ConfigField`` nests, so a template is deepest exactly where it is least read.
        """
        with pytest.raises(ConformanceError) as caught:
            check_tool(_tool_with_element_member_named(name))

        message = str(caught.value)
        assert "rules" in message and name in message, f"the refusal must path the member: {message}"

    @pytest.mark.parametrize("name", ["on_columns", "keys", "include", "period", "no_action", "yesterday", "only"])
    def test_an_ordinary_field_name_is_still_accepted(self, name: str) -> None:
        """Half the rule. One satisfied by refusing everything would be no rule at all.

        ``yesterday`` and ``only`` earn their place: a name that merely starts with an unsafe word
        is a different name, and refusing it would be a rule nobody could work with.
        """
        check_tool(_tool_with_field_named(name))
        check_tool(_tool_with_member_named(name))
        check_tool(_tool_with_element_member_named(name))

    def test_the_field_name_rule_and_the_enum_value_rule_refuse_the_same_words(self) -> None:
        """One published set, read by both rules.

        A second list typed out by hand is the defect this pins. The enum rule was moved into the
        catalog because three tools had each hand-rolled it, and all three missed the null
        spellings. A field-name rule with its own list would repeat that.
        """
        for word in sorted(YAML_UNSAFE_VALUES):
            with pytest.raises(ConformanceError):
                check_tool(_tool_with_field_named(word))

            with pytest.raises(ConformanceError):
                check_tool(_enum_tool(word))


# ---------------------------------------------------------------------------
# An input port name YAML would rewrite is a conformance failure
#
# An input port is named on the left of a colon in a pipeline file -- ``inputs: {rows: step.rows}``
# -- which is exactly where the rule above already applies to a config field name. A tool with an
# input port called ``on`` hands the loader the key ``True``. No tool has such a port today, so
# this is a hole rather than a bug. Output port names are a different case: a step refers to one as
# a value, on the right of the colon, where a loader leaves it alone.
# ---------------------------------------------------------------------------


def _tool_with_ports(input_name: str, output_name: str) -> Any:
    """A tool conforming in every other way, reading ``input_name`` and writing ``output_name``."""

    def execute(inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return Ports({output_name: inputs[input_name].with_columns(pl.lit(config["label"]).alias("label"))})

    return _StubTool(
        input_ports=(input_name,),
        output_ports=(output_name,),
        execute_fn=execute,
        examples=(
            WorkedExample(
                scenario="Label every row arriving on this step's one input port",
                configuration={"label": "x"},
                explanation="The rows arrive on the port this tool reads and leave on the port it writes.",
                inputs=Ports({input_name: pl.DataFrame({"id": [1]})}),
                outputs=Ports({output_name: pl.DataFrame({"id": [1], "label": ["x"]})}),
            ),
        ),
    )


def _tool_reading_a_port_named(name: str) -> Any:
    """The same tool, with only the input port name varying."""
    return _tool_with_ports(name, "out")


class TestAnInputPortNameYamlWouldRewrite:
    """A tool may not read a port whose name a definition file turns into something else.

    A step says what it reads as ``inputs: {rows: previous_step.rows}``. The port name is the key.
    A port called ``on`` gives the loader the key ``True``, and the name has stopped being a name --
    the same defect the field-name rule above was written for, one field over.

    Quoting it in the file works. It also puts the fix on every author of every pipeline using the
    tool, and nothing tells them. The tool has one author, so the refusal belongs on the tool.
    """

    @pytest.mark.parametrize("name", _unsafe_field_name_cases())
    def test_check_tool_refuses_an_input_port_name_yaml_would_rewrite(self, name: str) -> None:
        """Every word the published set holds, plus the bare numbers a file hands over as numbers."""
        with pytest.raises(ConformanceError) as caught:
            check_tool(_tool_reading_a_port_named(name))

        message = str(caught.value)
        assert "YAML" in message or "yaml" in message, f"the refusal must say why: {message}"
        if name.strip():
            assert name in message, f"the refusal must name the port it is about: {message}"

    @pytest.mark.parametrize("name", ["rows", "in", "left", "right", "on_hand", "no_match", "yesterday"])
    def test_an_ordinary_input_port_name_is_still_accepted(self, name: str) -> None:
        """Half the rule. One satisfied by refusing everything would be no rule at all.

        ``on_hand``, ``no_match`` and ``yesterday`` earn their place: a name that merely starts with
        an unsafe word is a different name, and a file hands it over as written.
        """
        check_tool(_tool_reading_a_port_named(name))

    def test_the_input_port_rule_and_the_field_name_rule_refuse_the_same_words(self) -> None:
        """One published set, read by both rules.

        A second list typed out by hand is the defect this pins. Three tools each hand-rolled this
        check once, and all three missed the null spellings.
        """
        for word in sorted(YAML_UNSAFE_VALUES):
            with pytest.raises(ConformanceError):
                check_tool(_tool_reading_a_port_named(word))

            with pytest.raises(ConformanceError):
                check_tool(_tool_with_field_named(word))

    def test_the_rule_is_about_the_input_side_only(self) -> None:
        """An output port keeps such a name, and the same tool with it on the input side is refused.

        The split is the whole ruling, so it is pinned in one place rather than inferred from two
        tests that pass for their own reasons. ``filter`` ships output ports named ``true`` and
        ``false``, and a step refers to one as a value, where a file leaves it alone.
        """
        check_tool(_tool_with_ports("rows", "false"))

        with pytest.raises(ConformanceError):
            check_tool(_tool_with_ports("false", "rows"))


# ---------------------------------------------------------------------------
# A config key that is not a name at all
#
# ``_check_level`` reports an unknown field by joining the keys it did not recognise. A key that is
# not a string raises TypeError from inside that join. So the one path that runs only when
# something is already wrong is the path that does not work. The rule above closes the route that
# found it, but a config reaches this code from a hand-built dict and from a generated definition
# too, and an error path that has never run against a real error is not an error path.
# ---------------------------------------------------------------------------


def _not_a_name_cases() -> list[Any]:
    """The keys a definition file hands over for an unquoted word or a bare number."""
    return [
        pytest.param(True, id="boolean_true"),
        pytest.param(False, id="boolean_false"),
        pytest.param(None, id="null"),
        pytest.param(3, id="integer"),
        pytest.param(1.5, id="float"),
    ]


def _template_with_an_object_and_an_array() -> dict[str, ConfigField]:
    """A template deep enough to say where a bad key was found."""
    return {
        "label": ConfigField(type="string", description="The label to add."),
        "options": ConfigField(
            type="object",
            description="Options for the step.",
            fields={"mode": ConfigField(type="string", description="How to add it.")},
        ),
        "rules": ConfigField(
            type="array",
            description="Rules applied in order.",
            items=ConfigField(
                type="object",
                description="One rule.",
                fields={"mode": ConfigField(type="string", description="How to add it.")},
            ),
        ),
    }


class TestAConfigKeyThatIsNotAName:
    """A config carrying a key that is not a string is refused, and the refusal is readable.

    Today it raises TypeError from inside the code that builds the error message. Nobody reading
    ``sequence item 0: expected str instance, bool found`` learns anything about their file.
    """

    @pytest.mark.parametrize("key", _not_a_name_cases())
    def test_a_key_that_is_not_a_string_is_refused_as_a_config_error(self, key: Any) -> None:
        """The refusal is a ConfigError. It is never a TypeError raised by the error reporter."""
        with pytest.raises(ConfigError):
            validate_against_template(_template_with_an_object_and_an_array(), {key: "x"})

    @pytest.mark.parametrize("key", _not_a_name_cases())
    def test_the_refusal_shows_the_key_and_what_it_is(self, key: Any) -> None:
        """``True`` and ``"True"`` print the same, so the type is what tells the two apart."""
        with pytest.raises(ConfigError) as caught:
            validate_against_template(_template_with_an_object_and_an_array(), {key: "x"})

        message = str(caught.value)
        assert repr(key) in message, f"the refusal must show the offending key: {message}"
        assert type(key).__name__ in message, f"the refusal must say what the key is: {message}"

    @pytest.mark.parametrize("key", _not_a_name_cases())
    def test_the_refusal_says_the_key_is_not_a_name_and_suggests_quoting_it(self, key: Any) -> None:
        """The fix in the file is one pair of quotes, so the message says so.

        It also must not call this an unknown field. That sends an author looking for a field they
        never wrote, when what they wrote was a field name the loader ate.
        """
        with pytest.raises(ConfigError) as caught:
            validate_against_template(_template_with_an_object_and_an_array(), {key: "x"})

        message = str(caught.value).lower()
        assert "name" in message, f"the refusal must say the key is not a name: {message}"
        assert "quot" in message, f"the refusal must point at the fix: {message}"
        assert "unknown" not in message, f"this is not an unknown field, and saying so misleads: {message}"

    def test_a_key_that_is_not_a_string_inside_an_object_is_refused_by_path(self) -> None:
        """A member of an object comes from the file as a key too, and the path locates it."""
        with pytest.raises(ConfigError) as caught:
            validate_against_template(_template_with_an_object_and_an_array(), {"options": {True: "x"}})

        message = str(caught.value)
        assert "options" in message, f"the refusal must say where the key was found: {message}"
        assert "True" in message and "bool" in message, f"the refusal must show the key: {message}"

    def test_a_key_that_is_not_a_string_inside_an_array_of_objects_is_refused_by_path(self) -> None:
        """An ordered list of objects is where a bad key is hardest to find by hand.

        Below the top level the join does not blow up -- the path is built with an f-string, which
        swallows anything -- so the failure changes shape rather than going away. The author is
        told they wrote an unknown field called ``rules[1].None``, which they did not.
        """
        with pytest.raises(ConfigError) as caught:
            validate_against_template(
                _template_with_an_object_and_an_array(),
                {"rules": [{"mode": "normal"}, {None: "x"}]},
            )

        message = str(caught.value)
        assert "rules" in message, f"the refusal must say where the key was found: {message}"
        assert "None" in message and "NoneType" in message, f"the refusal must show the key: {message}"
        assert "unknown" not in message.lower(), f"this is not an unknown field, and saying so misleads: {message}"
        assert "quot" in message.lower(), f"the refusal must point at the fix: {message}"

    def test_an_unknown_field_with_an_ordinary_name_is_still_reported_as_unknown(self) -> None:
        """The guard against the fix swallowing the refusal it sits in front of.

        A misspelled field is the common case, and it keeps its own message.
        """
        with pytest.raises(ConfigError) as caught:
            validate_against_template(_template_with_an_object_and_an_array(), {"labell": "x"})

        message = str(caught.value)
        assert "unknown" in message.lower(), f"a misspelled field is still an unknown field: {message}"
        assert "labell" in message, f"the refusal must name what it did not recognise: {message}"


# ---------------------------------------------------------------------------
# A declaration that constrains nothing
#
# The module already refuses ``items`` outside an array and members outside an object, for the
# stated reason that a silent no-op is the worst outcome available: a template that reads as if it
# constrains something and does not. Two more declarations read exactly that way.
# ---------------------------------------------------------------------------


def _inert_declaration_cases() -> list[Any]:
    """(template, substrings the refusal must carry) for a declaration that could never be applied."""
    return [
        pytest.param(
            {
                "label": ConfigField(type="string", description="The label to add.", required=True),
                "columns": ConfigField(
                    type="array",
                    description="Columns to write.",
                    items=ConfigField(
                        type="object",
                        description="One column.",
                        fields={
                            "name": ConfigField(type="string", description="Its name.", required=True),
                            "replace": ConfigField(
                                type="boolean",
                                description="Overwrite an existing column.",
                                default=False,
                            ),
                        },
                    ),
                ),
            },
            ("columns[].replace", "default"),
            id="a_default_on_a_member_of_an_element",
        ),
        pytest.param(
            {
                "label": ConfigField(type="string", description="The label to add.", required=True),
                "columns": ConfigField(
                    type="array",
                    description="Columns to write.",
                    items=ConfigField(type="string", description="One column name.", default="key"),
                ),
            },
            ("columns[]", "default"),
            id="a_default_on_an_element_itself",
        ),
        pytest.param(
            {
                "label": ConfigField(type="string", description="The label to add.", required=True),
                "columns": ConfigField(
                    type="array",
                    description="Columns to write.",
                    items=ConfigField(type="string", description="One column name.", required=True),
                ),
            },
            ("columns[]", "required"),
            id="required_on_the_element_spec_itself",
        ),
    ]


class TestADeclarationThatCouldNeverBeApplied:
    """``apply_defaults`` deliberately fills nothing inside an array or an object, so a nested
    default is inert -- yet conformance validates it, which reads as approval.

    A tool author declares ``default=False`` on ``columns[].replace``, conformance passes, the
    default is never applied, and ``execute`` still has to write ``entry.get("replace")``. The
    declaration is worse than absent, because the next reader trusts it. ``required`` on the
    ``items`` spec itself is the same shape: nothing reads it, and what the author meant --
    "elements must be there" -- is what ``non_empty`` on the array says.

    ``required`` on a *member* of an element is different and stays: it is enforced per element,
    which the second test pins so the rule cannot creep from the inert declarations into the live
    one.
    """

    @pytest.mark.parametrize(("template", "expected_substrings"), _inert_declaration_cases())
    def test_a_declaration_that_could_never_be_applied_is_a_conformance_failure(
        self,
        template: Mapping[str, ConfigField],
        expected_substrings: tuple[str, ...],
    ) -> None:
        """Refused where it is declared, by its path, the way an element type outside an array is."""
        with pytest.raises(ConformanceError) as caught:
            check_tool(_StubTool(template=template))

        message = str(caught.value)
        for substring in expected_substrings:
            assert substring in message, f"refusal should say {substring!r}: {message}"

    def test_a_default_at_the_top_level_and_required_inside_an_element_both_still_hold(self) -> None:
        """The two neighbouring declarations that *are* applied, so the rule cannot over-reach.

        A top-level default is filled by ``apply_defaults`` and a nested ``required`` is enforced
        per element by the validator -- and an ordered-rules tool depends on both.
        """
        check_tool(_CaseWhenTool())

        template = _case_when_template()
        assert apply_defaults(_permissive_template(), {"label": "x"})["mode"] == "append"

        with pytest.raises(ConfigError) as caught:
            validate_against_template(template, {"rules": [{"then": "'a'"}], "otherwise": "'z'", "output": "label"})

        assert "rules[0].when" in str(caught.value), f"required inside an element is enforced: {caught.value}"


# ---------------------------------------------------------------------------
# An output column that would silently replace an input one
#
# Four tools write a named output column, and all four had to answer the same question about a
# name that is already there. They did not answer it the same way. Refusing replacement outright is
# too strict -- an in-place repair is a legitimate step -- and taking the name silently is the real
# hazard, because whoever writes one step does not know all forty columns of the frame they are
# writing into. So the answer is a keystroke, and the decision is spelled once here.
# ---------------------------------------------------------------------------


def _output_column_cases() -> list[Any]:
    """(column name, the ``replace`` flag, substrings a refusal must carry or ``None`` if accepted)."""
    return [
        pytest.param("total", False, None, id="a_new_column_is_written"),
        pytest.param("total", None, None, id="an_unset_flag_is_not_a_request_to_replace"),
        pytest.param("code", True, None, id="an_in_place_repair_the_author_asked_for"),
        pytest.param("code", False, ("code", "output", "replace"), id="a_collision_the_author_never_typed"),
        pytest.param("code", None, ("code", "output"), id="a_collision_under_an_unset_flag"),
        pytest.param("cdoe", True, ("cdoe",), id="a_replacement_of_a_column_that_is_not_there"),
    ]


class TestASharedAnswerForAnOutputColumnName:
    """One mechanism, so four tools cannot drift into four answers.

    The flag stays in each tool's own template rather than being contributed by the contract,
    because its *path* differs: it is per-entry inside an array for a tool writing several columns
    and top-level for the three writing one, and a contract that injected a field would have to
    rewrite a mapping the tool returned -- making ``set(template)`` unpredictable for every test and
    every rendered catalog page. What the contract contributes instead is both halves that *can* be
    shared: the decision, as one function with one message, and the canonical declaration, so the
    field reads the same on all four pages.
    """

    @pytest.mark.parametrize(("name", "replace", "expected_substrings"), _output_column_cases())
    def test_a_silent_replacement_is_refused_and_so_is_a_replacement_of_nothing(
        self,
        name: str,
        replace: bool | None,
        expected_substrings: tuple[str, ...] | None,
    ) -> None:
        """Both directions, because a misspelling under ``replace: true`` is the subtler defect.

        Unchecked, it quietly creates a new column while the column the author meant to repair
        keeps its old values -- a run that reports success and changes nothing.
        """
        frame = _typed_frame()

        if expected_substrings is None:
            assert check_output_column(frame, name, "output", replace=replace) is None
            return

        with pytest.raises(ConfigError) as caught:
            check_output_column(frame, name, "output", replace=replace)

        message = str(caught.value)
        for substring in expected_substrings:
            assert substring in message, f"refusal should say {substring!r}: {message}"

    def test_the_replace_flag_is_declared_once_and_is_legal_inside_an_element(self) -> None:
        """The canonical declaration carries no default, which is what makes it legal nested.

        A default here would be inert inside an array -- refused by the rule above -- so the flag
        means "not asked" when absent, which is the reading ``check_output_column`` takes.
        """
        assert isinstance(REPLACE_FIELD, ConfigField)
        assert REPLACE_FIELD.type == "boolean", "a flag YAML carries as written"
        assert REPLACE_FIELD.default is None, "a default on this field is never applied inside an array"
        assert REPLACE_FIELD.required is False
        assert REPLACE_FIELD.description.strip(), "it is a catalog page, not only a validator"

        check_tool(
            _StubTool(
                template={
                    "label": ConfigField(type="string", description="The label to add.", required=True),
                    "replace": REPLACE_FIELD,
                    "columns": ConfigField(
                        type="array",
                        description="Columns to write.",
                        items=ConfigField(
                            type="object",
                            description="One column.",
                            fields={
                                "name": ConfigField(type="string", description="Its name.", required=True),
                                "replace": REPLACE_FIELD,
                            },
                        ),
                    ),
                }
            )
        )


# ---------------------------------------------------------------------------
# An output column name that is not a name at all
#
# The same question one step earlier than the one above: not "may this name replace that column",
# but "is this a name". Seven tools were asked it by measurement and gave three different answers --
# refused both shapes, refused the empty one and accepted three spaces, accepted both -- and
# ``case_when`` under ``replace: true`` gave a fourth, refusing a blank as a column the input does
# not carry, which sends the author to look at their frame when the fault is in their file.
#
# So the rule belongs here, in the seam every one of them already reaches, and ahead of both
# questions that seam already asks: a blank is wrong whether or not the flag is set, and it is
# wrong before anything about the frame is consulted. A tool added next year inherits the answer
# by calling the seam, which it must call anyway.
#
# Blank means empty or nothing but whitespace. A column called "   " is invisible in every
# rendering, sits in a header row indistinguishable from its neighbours, and cannot be written in
# an expression at all, because "[   ]" is not something an author can type and mean.
# ---------------------------------------------------------------------------

# Every shape of nothing a definition file can put where a name belongs. A file that lost its value
# leaves "", a file whose value was indented away leaves spaces, and a file edited with a tab key
# leaves "\t" -- each arrives as a string that passes every type check there is.
#
# The last four are the ones that matter most and are the easiest to leave out. A header cell copied
# out of a spreadsheet or lifted from a web page is a NON-BREAKING space far more often than a plain
# one, and it is the single blank an author cannot see in their own file. They are here because the
# rule is "nothing that Python calls whitespace", not "no spaces or tabs" -- and narrowing it to the
# ASCII four is a one-line change that every other test in this file would let through.
_BLANK_NAMES: Final[tuple[str, ...]] = (
    "",
    " ",
    "   ",
    "\t",
    "\n",
    " \t ",
    "\xa0",  # non-breaking space, what a spreadsheet copy actually gives you
    "\u3000",  # ideographic space
    "\x85",  # next line
    "\xa0 \t",  # mixed, so a rule that checks only the first character is caught
)

# Names that are odd and still names. This half is what stops the rule overreaching: it is about a
# name that is *nothing*, never a name that is merely unusual, and every one of these is a column
# an author may legitimately want and a downstream step can legitimately reference.
# The last two are a decision, not an oversight. A zero-width space and a byte order mark are
# invisible and Python does not call them whitespace, so they are ACCEPTED and write a column nobody
# can see. The line is drawn at str.isspace() deliberately: any other line is a second definition of
# "blank" with no published home, and each of these is vanishingly rare beside the non-breaking
# space above. They are pinned here so that moving the line is an argument with a test rather than a
# quiet tidy.
_UNUSUAL_BUT_REAL_NAMES: Final[tuple[str, ...]] = (
    " total ",
    "x",
    "9",
    "a-b",
    "TOTAL",
    "_",
    "日付",
    "\u200b",
    "\ufeff",
)

# Phrases belonging to the seam's *other* refusals. A blank name must never be reported with any of
# them: "the input does not have it" is true of a blank and is not what went wrong.
_ABSENT_COLUMN_PHRASES: Final[tuple[str, ...]] = (
    "asks to replace",
    "does not have",
    "already carries",
)


def _blank_seam_cases() -> list[Any]:
    """Every blank shape against every reading of the flag, since none of them may differ."""
    return [
        pytest.param(name, replace, id=f"{name!r}_with_replace_{replace}")
        for name in _BLANK_NAMES
        for replace in (None, False, True)
    ]


class TestABlankOutputColumnNameIsRefusedByTheSharedSeam:
    """A name made of nothing is refused here, once, for every tool that asks.

    Put in the seam rather than in the template's ``non_empty``, for two reasons. ``non_empty`` is
    declared per field, so a tool whose author did not declare it -- ``case_when``, today -- gets no
    answer at all, which is exactly how the catalog came to hold three of them. And ``non_empty``
    knows nothing about ``replace``, whereas rule four says the blank must be ruled on *before* the
    flag is read; the seam is the only place that sees both.
    """

    @pytest.mark.parametrize(("name", "replace"), _blank_seam_cases())
    def test_a_name_made_of_nothing_is_refused_whatever_the_flag_says(
        self,
        name: str,
        replace: bool | None,
    ) -> None:
        """Naming the field, saying what a name must be, and never as an absent column.

        The absent-column wording is the one an author gets today under ``replace: true``, and it
        is actively misleading: it describes the frame, and the frame is not what is wrong.
        """
        with pytest.raises(ConfigError) as caught:
            check_output_column(_typed_frame(), name, "output", replace=replace)

        message = str(caught.value)
        assert "'output'" in message, f"the refusal must name the field it is about: {message}"
        assert "blank" in message, f"the refusal must say what is wrong: {message}"
        assert "whitespace" in message, (
            f"the refusal must say what a name has to be, or an author looking at three spaces in "
            f"a file cannot see why it was refused: {message}"
        )
        for phrase in _ABSENT_COLUMN_PHRASES:
            assert phrase not in message, f"a blank name is not an absent column: {message}"

    def test_the_refusal_is_one_sentence_with_one_substitution_in_it(self) -> None:
        """The same words for every blank shape, every reading of the flag, and every field.

        One fault and one fix, so one sentence. The offending value is deliberately not quoted
        back: a name made of whitespace prints indistinguishably from an empty one, so repeating it
        tells the author nothing, and quoting it is exactly what would fork the wording six ways.
        The only thing a reader may see change is the address, and ``columns[0].name`` -- the
        nested path a tool writing several columns reports -- has to read as the same refusal as
        the top-level ``output`` a tool writing one reports.
        """
        frame = _typed_frame()
        refusals = set()
        for name in _BLANK_NAMES:
            for replace in (None, False, True):
                with pytest.raises(ConfigError) as caught:
                    check_output_column(frame, name, "output", replace=replace)
                refusals.add(str(caught.value))

        assert len(refusals) == 1, f"one rule, one wording; got {len(refusals)}: {sorted(refusals)}"

        with pytest.raises(ConfigError) as nested:
            check_output_column(frame, "", "columns[0].name", replace=True)

        rewritten = refusals.pop().replace("'output'", "'columns[0].name'")
        assert rewritten == str(nested.value), (
            f"the two refusals must be one sentence with one substitution:\n  {rewritten}\n  {nested.value}"
        )

    @pytest.mark.parametrize("name", _UNUSUAL_BUT_REAL_NAMES)
    def test_a_name_that_is_merely_unusual_is_still_a_name(self, name: str) -> None:
        """Leading and trailing space around real text, one character, a digit, punctuation, a
        non-Latin name: each is a column an author may want and none of them is nothing.

        ``" total "`` is the case that decides it. The rule may not trim, tidy or second-guess a
        name -- only refuse one that has no characters in it at all.
        """
        assert check_output_column(_typed_frame(), name, "output", replace=None) is None

    def test_a_frame_with_no_rows_answers_exactly_the_same(self) -> None:
        """This is a question about the configuration, so the data may not change the answer.

        An empty frame is what a nightly run hands the step after a filter matched nothing, and a
        rule that only fires on rows would let that run write a nameless column in silence.
        """
        empty = _typed_frame().head(0)
        assert empty.height == 0, "the frame under test must genuinely carry no rows"

        with pytest.raises(ConfigError) as on_empty:
            check_output_column(empty, "   ", "output", replace=None)
        with pytest.raises(ConfigError) as on_rows:
            check_output_column(_typed_frame(), "   ", "output", replace=None)

        assert str(on_empty.value) == str(on_rows.value)
        assert check_output_column(empty, "total", "output", replace=None) is None


# ---------------------------------------------------------------------------
# The same rule, stated over the shipped catalog rather than over the seam
#
# The point of putting the rule in one place is that nobody has to remember it. So the statement
# below is made over every tool that names an output column, and the roster it runs on is checked
# against the registry rather than trusted: a tool added next year that calls the seam and is not
# listed here fails the completeness test, which is the only way the coverage cannot go stale
# quietly.
# ---------------------------------------------------------------------------


def _catalog_frame() -> pl.DataFrame:
    """One frame every output-naming tool in the catalog can be run against."""
    return pl.DataFrame(
        {
            "code": ["a", "b"],
            "count": [1, 2],
            "year": [2024, 2024],
            "month": [1, 2],
        }
    ).with_columns(
        pl.date(pl.col("year"), pl.col("month"), 1).alias("opened"),
        pl.date(pl.col("year"), pl.col("month"), 28).alias("closed"),
    )


# Each entry: the tool's registered name, the config path that names its output column, and a
# builder putting ``name`` at that path in an otherwise valid configuration. Hardcoded because a
# valid configuration cannot be invented from a template, and checked for completeness below.
_OUTPUT_NAMING_TOOLS: Final[Mapping[str, tuple[str, Callable[[str], Mapping[str, Any]]]]] = {
    "formula": ("columns[0].name", lambda name: {"columns": [{"name": name, "value": "[count] + 1"}]}),
    "assemble_date": ("output", lambda name: {"year": "[year]", "month": "[month]", "output": name}),
    "date_diff": (
        "output",
        lambda name: {"start": "[opened]", "end": "[closed]", "unit": "day", "counting": "inclusive", "output": name},
    ),
    "record_id": ("output", lambda name: {"output": name}),
    "normalize_text": (
        "output",
        lambda name: {"source": "[code]", "output": name, "operations": ["lower"]},
    ),
    "case_when": (
        "output",
        lambda name: {"cases": [{"when": "[count] > 1", "then": "'big'"}], "output": name},
    ),
    "regex_extract_groups": (
        "position_column",
        lambda name: {"source": "[code]", "pattern": r"(?<letter>[a-z])", "position_column": name},
    ),
}


def _read_excel_config(name: str, directory: Path) -> Mapping[str, Any]:
    """Plant a two-sheet workbook in ``directory`` and stack both sheets, naming the sheet column ``name``."""
    path = directory / "months.xlsx"
    with xlsxwriter.Workbook(str(path)) as workbook:
        for sheet, key in (("Jan 2026", "j1"), ("Feb 2026", "f1")):
            worksheet = workbook.add_worksheet(sheet)
            worksheet.write_row(0, 0, ["key", "amount"])
            worksheet.write_row(1, 0, [key, 1.5])
    return {"path": str(path), "sheets_matching": "2026", "sheet_column": name}


# Source tools read no input port, so the frame above cannot reach them. Each entry here writes the
# file its configuration reads into the test's own directory, and the tool runs with no inputs.
# Otherwise the same as the roster above, and checked for completeness together with it.
_SOURCE_OUTPUT_NAMING_TOOLS: Final[Mapping[str, tuple[str, Callable[[str, Path], Mapping[str, Any]]]]] = {
    "read_excel": ("sheet_column", _read_excel_config),
}


def _roster() -> frozenset[str]:
    """Every tool either roster covers."""
    return frozenset(_OUTPUT_NAMING_TOOLS) | frozenset(_SOURCE_OUTPUT_NAMING_TOOLS)


def _run_naming(tool_name: str, name: str, directory: Path) -> Ports:
    """Run one roster tool with ``name`` as its output column.

    A tool that reads rows runs on the shared frame. A source tool runs on the file its builder
    writes into ``directory``, with no inputs.
    """
    tool = REGISTRY.get(tool_name)
    if tool_name in _SOURCE_OUTPUT_NAMING_TOOLS:
        _path, build_from_file = _SOURCE_OUTPUT_NAMING_TOOLS[tool_name]
        return run_tool(tool, Ports({}), build_from_file(name, directory))
    _path, build = _OUTPUT_NAMING_TOOLS[tool_name]
    return run_tool(tool, Ports({"rows": _catalog_frame()}), build(name))


def _catalog_blank_cases() -> list[Any]:
    """Every tool in either roster against every shape of nothing."""
    return [
        pytest.param(tool_name, name, id=f"{tool_name}_given_{name!r}")
        for tool_name in sorted(_roster())
        for name in _BLANK_NAMES
    ]


def _tools_calling_the_seam() -> frozenset[str]:
    """The registered tools whose own code calls :func:`check_output_column`.

    Read off the syntax rather than the text, so a tool that only *mentions* the seam in prose --
    ``cross_tab`` does, to explain why its own question is a different one -- is not counted.
    """

    calling: set[str] = set()
    for registered in REGISTRY.tools():
        try:
            source = inspect.getsource(type(registered))
        except (OSError, TypeError):  # pragma: no cover -- every shipped tool has source
            continue
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "check_output_column":
                    calling.add(registered.name)
    return frozenset(calling)


class TestEveryToolThatNamesAnOutputColumnRefusesABlankOne:
    """The catalog-wide statement, so the seventh tool is covered before anybody writes it."""

    @pytest.mark.parametrize(("tool_name", "name"), _catalog_blank_cases())
    def test_a_blank_name_never_reaches_the_engine(self, tool_name: str, name: str, tmp_path: Path) -> None:
        """Refused, by whichever of its rules gets there first.

        Wording is deliberately not asserted here. Four of these tools answered this correctly
        before the shared rule existed and answer it in their own words, which must not change;
        what this test pins is that no tool in the catalog writes a column with no name.
        """
        with pytest.raises(ConfigError) as caught:
            _run_naming(tool_name, name, tmp_path)

        assert "config field" in str(caught.value), f"a configuration fault, reported as one: {caught.value}"

    @pytest.mark.parametrize("tool_name", sorted(_roster()))
    def test_the_same_tools_still_take_a_name_that_is_merely_unusual(self, tool_name: str, tmp_path: Path) -> None:
        """The other half, tool by tool: a real name with spaces around it is written as spelled."""
        result = _run_naming(tool_name, " total ", tmp_path)

        assert " total " in result["rows"].columns, f"{tool_name} must write the name as spelled"

    def test_the_roster_is_every_tool_that_reaches_the_shared_seam(self) -> None:
        """The guard on the list above, so coverage cannot go stale in silence.

        A tool added later that names an output column calls the seam -- that is how it gets the
        collision rule it needs anyway -- and the moment it does, this fails until somebody adds it
        to the roster and it is covered by every case above.
        """
        assert _tools_calling_the_seam() == _roster(), (
            f"the roster and the catalog disagree; only in the roster: "
            f"{sorted(_roster() - _tools_calling_the_seam())}, only in the "
            f"catalog: {sorted(_tools_calling_the_seam() - _roster())}"
        )
        reads_rows = sorted(name for name in _SOURCE_OUTPUT_NAMING_TOOLS if REGISTRY.get(name).input_ports)
        assert not reads_rows, f"listed as source tools but they read input ports: {reads_rows}"


# ---------------------------------------------------------------------------
# A hook for "the configuration against the input schemas"
#
# The boundary checks the template and the tool adds its cross-field rules, but nothing owns the
# question in between: do the columns this configuration names exist on the ports it will read?
# Three tools already answer it first thing in ``execute``, in three different sentences. The
# fourth wording is where they disagree, and a tool that forgets to ask reaches the engine.
# ---------------------------------------------------------------------------


class _OrderSpyTool(_StubTool):
    """A conforming tool that records which boundary phases it was asked for, in order."""

    def __init__(self, *, refuse_inputs: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._refuse_inputs = refuse_inputs
        self.calls: list[str] = []

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self.calls.append("validate_config")
        super().validate_config(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        self.calls.append("validate_inputs")
        if self._refuse_inputs:
            require_columns(inputs["in"], ["cdoe"], "label", "in")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self.calls.append("execute")
        return super().execute(inputs, config)


class _ToolWithoutInputValidation:
    """Every member of the protocol except the new hook -- a tool nobody has migrated yet.

    Written out rather than derived from the stub, because inheriting would inherit the hook and
    the point is its absence: the registry's gate is attribute presence, so a missing member has to
    be caught there rather than discovered when the boundary calls it.
    """

    name = "unmigrated"
    description = "Add a fixed label column to every row."
    category = "transform"
    input_ports = ("in",)
    output_ports = ("out",)

    def __init__(self) -> None:
        self._inner = _StubTool(name="unmigrated")

    def config_template(self) -> Mapping[str, ConfigField]:
        return self._inner.config_template()

    def help(self) -> ToolHelp:
        return self._inner.help()

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self._inner.validate_config(config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        return self._inner.execute(inputs, config)


def _copy_the_named_column(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """Copy the column the configuration names -- work that needs the column to be there."""
    frame = inputs["in"]
    return Ports({"out": frame.with_columns(frame[config["label"]].alias("copied"))})


class _ColumnNamingTool(_StubTool):
    """A tool whose configuration names an input column, checked in the hook rather than in run."""

    def __init__(self, documented: str) -> None:
        super().__init__(
            name="column_naming",
            execute_fn=_copy_the_named_column,
            examples=(
                WorkedExample(
                    scenario="Copy the column the configuration names",
                    configuration={"label": documented},
                    explanation="The named column is copied into 'copied', so it has to exist.",
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"out": pl.DataFrame({"id": [1], "copied": [1]})}),
                ),
            ),
        )
        self.ran: list[str] = []

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        require_columns(inputs["in"], [config["label"]], "label", "in")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self.ran.append(config["label"])
        return super().execute(inputs, config)


def _require_columns_cases() -> list[Any]:
    """(names asked for, the absent one a refusal must name, or ``None`` where all are present)."""
    return [
        pytest.param(["key", "code"], None, id="every_column_is_there"),
        pytest.param([], None, id="nothing_is_asked_for"),
        pytest.param(["cdoe"], "cdoe", id="a_misspelled_column"),
        pytest.param(["key", "stat"], "stat", id="one_absent_among_present_ones"),
        pytest.param(["KEY"], "KEY", id="a_column_whose_case_is_wrong"),
    ]


class TestTheConfigurationAgainstTheInputSchemas:
    """A third question, between the template and the tool's own rules, with nowhere to live.

    The template cannot answer it -- it has no frame -- and a tool's ``validate_config`` cannot
    either, for the same reason. So it lands first thing in ``execute``, where each tool words it
    for itself and where the refusal arrives after the boundary has already said the tool was
    running. Given a hook, ``check_tool`` gains something real too: an example whose configuration
    disagrees with its own example inputs stops being an execution failure and becomes what it is,
    a configuration that was never valid.
    """

    def test_the_hook_runs_after_the_config_and_before_execute_and_can_refuse_first(self) -> None:
        """Its place in the order is the contract: a column that is not there stops the run early.

        And a stray exception from inside it says which phase it was, like every other piece of
        tool-authored code the boundary calls -- a reader sent to "validating its configuration"
        opens the wrong function.
        """
        conforming = _OrderSpyTool()
        run_tool(conforming, Ports({"in": _frame()}), {"label": "x"})
        assert conforming.calls == ["validate_config", "validate_inputs", "execute"]

        refusing = _OrderSpyTool(refuse_inputs=True)
        with pytest.raises(ConfigError) as caught:
            run_tool(refusing, Ports({"in": _frame()}), {"label": "x"})
        assert "cdoe" in str(caught.value)
        assert refusing.calls == ["validate_config", "validate_inputs"], "execute must never have run"

        cause = KeyError("label")
        phases = {}
        for phase in ("validate_config", "validate_inputs"):
            with pytest.raises(ExecutionError) as broke:
                run_tool(_BrokenPhaseTool(phase, cause), Ports({"in": _frame()}), {"label": "x"})
            phases[phase] = str(broke.value)
            assert "phase_trouble" in phases[phase] and "KeyError" in phases[phase], phases[phase]
        assert phases["validate_config"] != phases["validate_inputs"], f"the two phases read alike: {phases}"

    def test_a_tool_that_has_no_such_hook_is_refused_rather_than_quietly_accepted(self) -> None:
        """The one breaking change here: a member added to the protocol un-conforms every tool.

        That is the price of the gate being real. A boundary that called the hook only where it
        happened to exist would be a rule that holds when an author remembers, which is the
        convention this module keeps replacing with a rule.
        """
        unmigrated = _ToolWithoutInputValidation()

        assert not isinstance(unmigrated, Tool), "the protocol gate is attribute presence"

        with pytest.raises(RegistryError) as caught:
            ToolRegistry().register(unmigrated)
        assert "Tool protocol" in str(caught.value)
        assert type(unmigrated).__name__ in str(caught.value), f"the refusal must identify it: {caught.value}"

        with pytest.raises(ConformanceError):
            check_tool(unmigrated)

    @pytest.mark.parametrize(("names", "absent"), _require_columns_cases())
    @pytest.mark.parametrize("height", [3, 0], ids=["a_frame_with_rows", "a_frame_with_no_rows"])
    def test_require_columns_names_the_absent_the_field_the_port_and_what_is_present(
        self,
        names: Sequence[str],
        absent: str | None,
        height: int,
    ) -> None:
        """The port is the reason a shared version beats each tool writing its own.

        A one-input tool passes ``rows``; a pairing tool passes ``left`` or ``right``, and "column
        not found" without saying which side is a refusal that sends the reader to both. Listing
        the columns that *are* present is what turns the refusal into the answer -- a misspelling
        and a wrong case are the two mistakes here, and both are obvious next to the real list.
        """
        frame = _typed_frame().head(height)

        if absent is None:
            assert require_columns(frame, names, "key_columns", "left") is None
            return

        with pytest.raises(ConfigError) as caught:
            require_columns(frame, names, "key_columns", "left")

        message = str(caught.value)
        assert absent in message, f"the refusal must name the absent column: {message}"
        assert "key_columns" in message, f"the refusal must name the config field: {message}"
        assert "left" in message, f"the refusal must name the port: {message}"
        for present in frame.columns:
            assert present in message, f"the refusal must list what is present, missing {present!r}: {message}"

    def test_an_example_whose_config_disagrees_with_its_own_inputs_is_a_configuration_defect(self) -> None:
        """Two defects a single "does not run" would report identically, with two different fixes.

        A documented example is the thing somebody copies, and one naming a column its own inputs
        lack is a configuration nobody could have run. Reported as a run failure it reads as if the
        tool were broken; reported as a configuration failure it points at the line to change.
        """
        check_tool(_ColumnNamingTool("id"))

        drifted = _ColumnNamingTool("di")
        with pytest.raises(ConformanceError) as caught:
            check_tool(drifted)

        message = str(caught.value)
        assert "di" in message, f"the refusal must name the column: {message}"
        assert "does not validate" in message, f"a configuration defect, not a run failure: {message}"
        assert drifted.ran == [], "an example that cannot be valid must never reach the engine"


# ---------------------------------------------------------------------------
# An example declares the world it needs
#
# Thirteen tools are total functions -- frames in, frames out, no environment and no clock -- and
# their worked examples run under ``check_tool`` with nothing prepared, which is exactly what makes
# the catalog trustworthy: every example on the generated page has been executed against its tool.
# A source or a sink breaks that assumption, and the only seam offered before this section was
# ``help()`` itself: build a database and export a variable while describing yourself. That is
# refused here, because the catalog (``catalog.catalog()``) calls ``help()`` for every registered
# tool, so generating the documentation would create databases and write to ``os.environ`` as a
# side effect of asking each tool what it does.
#
# So an example declares its world instead: one optional field holding a callable that returns a
# context manager, entered immediately before the example is checked and exited immediately after,
# in a ``try``/``finally``. A context manager rather than a setup/teardown pair, because the
# pairing is then impossible to get half-right. The declaration sits on the example, where a reader
# of the catalog looks, and ``help()`` goes back to being a description.
#
# Two disturbances are refused outright, on both sides of that seam: a ``help()`` that changes the
# environment or the working directory, and a setup whose teardown does not put them back. Neither
# rule can catch an arbitrary side effect -- a file written, a socket opened, a module global
# rewritten -- and neither pretends to. They catch the two that are realistic here and cheap to
# see, and a rule that catches the realistic mistake beats one that catches none.
# ---------------------------------------------------------------------------


# The variable a declared setup exports, and a value only a prepared world could have produced.
_PREPARED = "ETL_TEST_EXAMPLE_WORLD_CONNECTION"
_PREPARED_VALUE = "prepared-by-the-example"


@pytest.fixture
def process_left_as_found() -> Iterator[None]:
    """Snapshot the environment and the working directory, and put both back.

    Several cases below deliberately build a tool that leaks one or the other, and a leak that
    escaped this fixture would poison every later test in the same process -- which is the very
    failure the rules under test exist to refuse.
    """
    environment = dict(os.environ)
    directory = os.getcwd()
    yield
    os.environ.clear()
    os.environ.update(environment)
    os.chdir(directory)


def _prepared_world(events: list[str], *, leak: bool = False) -> Callable[[], AbstractContextManager[None]]:
    """A setup that records its own entry and exit, exports the variable, and refuses to nest.

    ``leak=True`` is the mistake: a teardown that runs and puts nothing back.
    """
    depth = [0]

    @contextmanager
    def prepare() -> Iterator[None]:
        depth[0] += 1
        if depth[0] > 1:
            raise AssertionError("a setup was entered again before the previous one was torn down")
        events.append("enter")
        os.environ[_PREPARED] = _PREPARED_VALUE
        try:
            yield
        finally:
            events.append("exit")
            depth[0] -= 1
            if not leak:
                os.environ.pop(_PREPARED, None)

    return prepare


@contextmanager
def _leaves_the_directory_moved() -> Iterator[None]:
    """A setup whose teardown leaves the process somewhere else -- the other realistic leak."""
    os.chdir(os.sep)
    yield


class _WorldReadingTool(_StubTool):
    """A tool that cannot validate or run without the world its example declares.

    Every phase the boundary calls records itself and refuses when the variable is absent, so the
    event list below is a claim about *coverage* as well as about order: a setup entered around
    ``run_tool`` alone would leave ``validate_config`` looking at an empty environment, which is
    where a source resolves its connection.
    """

    def __init__(
        self,
        events: list[str],
        setup: Callable[[], AbstractContextManager[None]] | None,
        *,
        documented: str = _PREPARED_VALUE,
        raises: bool = False,
    ) -> None:
        super().__init__(
            name="world_reading",
            examples=(
                WorkedExample(
                    scenario="Read the world the example prepared",
                    configuration={"label": "x"},
                    explanation="The label column holds what the declared setup put in the environment.",
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"out": pl.DataFrame({"id": [1], "label": [documented]})}),
                    setup=setup,
                ),
            ),
        )
        self.events = events
        self._raises = raises

    def _phase(self, phase: str) -> None:
        self.events.append(phase)
        if _PREPARED not in os.environ:
            raise ConfigError(f"{phase} ran before the world the example declared was prepared")

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self._phase("validate_config")
        super().validate_config(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        self._phase("validate_inputs")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self._phase("execute")
        if self._raises:
            raise RuntimeError("the engine fell over")
        return Ports({"out": inputs["in"].with_columns(pl.lit(os.environ[_PREPARED]).alias("label"))})


class _DisturbingHelpTool(_StubTool):
    """A tool whose ``help()`` prepares a world instead of describing one."""

    def __init__(self, disturb: Callable[[], None]) -> None:
        super().__init__()
        self._disturb = disturb

    def help(self) -> ToolHelp:
        self._disturb()
        return super().help()


def _leaky_setup_tool(setup: Callable[[], AbstractContextManager[None]]) -> tuple[Any, list[Mapping[str, Any]]]:
    """A tool whose example runs and compares clean, and whose setup puts nothing back."""
    seen, execute_fn = _recorder()
    tool = _StubTool(
        execute_fn=execute_fn,
        examples=(
            WorkedExample(
                scenario="Pass the rows through",
                configuration={"label": "x"},
                explanation="The rows arrive unchanged; the setup is the only interesting part.",
                inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                outputs=Ports({"out": pl.DataFrame({"id": [1]})}),
                setup=setup,
            ),
        ),
    )
    return tool, seen


def _disturbing_tool(kind: str) -> tuple[Any, list[Mapping[str, Any]]]:
    """One tool per way of disturbing the process, and the list its example's runs land in."""
    if kind == "help_exports_a_variable":
        return _DisturbingHelpTool(lambda: os.environ.update({_PREPARED: _PREPARED_VALUE})), []
    if kind == "help_moves_the_working_directory":
        return _DisturbingHelpTool(lambda: os.chdir(os.sep)), []
    if kind == "a_teardown_that_leaves_a_variable_behind":
        return _leaky_setup_tool(_prepared_world([], leak=True))
    return _leaky_setup_tool(_leaves_the_directory_moved)


def _disturbance_cases() -> list[Any]:
    """(how the process is disturbed, what a refusal must name, whether the example got to run)."""
    return [
        pytest.param("help_exports_a_variable", _PREPARED, False, id="help_exports_a_variable"),
        pytest.param("help_moves_the_working_directory", "directory", False, id="help_moves_the_directory"),
        pytest.param("a_teardown_that_leaves_a_variable_behind", _PREPARED, True, id="a_teardown_leaks_a_variable"),
        pytest.param("a_teardown_that_leaves_the_directory_moved", "directory", True, id="a_teardown_leaks_a_chdir"),
    ]


class TestAnExampleDeclaresTheWorldItNeeds:
    """The seam that documents a source or a sink without ``help()`` doing the preparing."""

    def test_the_setup_is_entered_before_every_phase_of_the_example_and_exited_after(
        self,
        process_left_as_found: None,
    ) -> None:
        """Enter, the whole check of the example, exit -- in that order and covering all of it.

        The order is the contract and the coverage is the part that is easy to get half-right: a
        setup entered around ``run_tool`` alone would leave both validation phases looking at a
        world that is not there, and resolving a connection before a row is read is precisely what
        a source does in ``validate_config``. Nothing may be left behind either, because the next
        example is checked in the same process.

        Order of first appearance, because a conformance check already validates an example once on
        its own account and once inside the run, and how many times it does that is not this
        seam's business.
        """
        events: list[str] = []
        tool = _WorldReadingTool(events, _prepared_world(events))

        check_tool(tool)

        assert list(dict.fromkeys(events)) == ["enter", "validate_config", "validate_inputs", "execute", "exit"]
        assert _PREPARED not in os.environ, "the declared world must not outlive the example"

    @pytest.mark.parametrize(
        ("documented", "raises", "refusal"),
        [
            pytest.param("something else", False, "drifted", id="the_example_drifts"),
            pytest.param(_PREPARED_VALUE, True, "does not run", id="the_tool_raises"),
        ],
    )
    def test_the_teardown_runs_even_when_the_example_fails_or_the_tool_raises(
        self,
        documented: str,
        raises: bool,
        refusal: str,
        process_left_as_found: None,
    ) -> None:
        """A red conformance check must not leave a database or a variable behind.

        The two ways a check ends badly are a comparison that fails and a tool that throws, and a
        teardown skipped in either case poisons every later check in the same process -- so the
        failure that finally gets reported is somebody else's.
        """
        events: list[str] = []
        tool = _WorldReadingTool(events, _prepared_world(events), documented=documented, raises=raises)

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        assert refusal in str(caught.value)
        assert events[0] == "enter", f"prepared first: {events}"
        assert events[-1] == "exit", f"torn down last, whatever happened in between: {events}"
        assert _PREPARED not in os.environ, "a failed check must leave the process as it found it"

    def test_the_same_example_can_be_checked_again_in_the_same_process(
        self,
        process_left_as_found: None,
    ) -> None:
        """``check_registry`` checks every tool, and a catalog is checked more than once a run.

        Twice through, one enter and one exit each time, and the second check sees exactly what the
        first did -- the property a sink's documented examples already depend on, since a setup
        still standing from the last check would make the next write land on a table that was
        supposed to be fresh. The setup refuses to be entered while still open, so an
        implementation that nested them fails here rather than mysteriously later.
        """
        events: list[str] = []
        registry = ToolRegistry()
        registry.register(_WorldReadingTool(events, _prepared_world(events)))

        check_registry(registry)
        check_registry(registry)

        assert events.count("enter") == 2, f"prepared once per check: {events}"
        assert events.count("exit") == 2, f"torn down once per check: {events}"
        half = len(events) // 2
        assert events[:half] == events[half:], f"the second check saw something else: {events}"
        assert events[half - 1] == "exit", f"each check tears down before the next prepares: {events}"
        assert _PREPARED not in os.environ

    @pytest.mark.parametrize(("kind", "named", "example_ran"), _disturbance_cases())
    def test_a_tool_that_disturbs_the_process_around_its_own_documentation_is_refused(
        self,
        kind: str,
        named: str,
        example_ran: bool,
        process_left_as_found: None,
    ) -> None:
        """The narrowest enforceable reading of "``help()`` is a description", and its mirror.

        A changed ``os.environ`` and a changed working directory are the two disturbances that
        actually happen here and are cheap to see, on both sides of the seam: a ``help()`` that
        prepares a world, and a teardown that puts nothing back. A file written, a connection
        opened or a module global rewritten still escapes, and nothing here claims otherwise --
        detecting an arbitrary side effect is not on offer.

        The two halves are told apart by *when* the refusal lands rather than by its wording. A
        disturbing ``help()`` is refused before any example runs, because the description is what
        is wrong. A leaking teardown is refused after its example has run and compared clean,
        because the example was fine and the tidying up was not.
        """
        tool, seen = _disturbing_tool(kind)

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert tool.name in message, f"a refusal must name the tool it came from: {message}"
        assert named in message, f"a refusal must name what moved: {message}"
        assert bool(seen) is example_ran, f"the example ran: {bool(seen)}, expected {example_ran}"

    def test_an_example_that_declares_nothing_is_checked_exactly_as_before(self) -> None:
        """The fence: the declaration is optional, and only a source or a sink may need one.

        A change that made the declaration mandatory -- or that hung one on an example that needs
        no world -- would be a change to what the catalog documents rather than to how it is
        checked. The rule is stated by category rather than by listing today's tools, because the
        list was already out of date once: this test originally asserted that *no* shipped example
        declared a world, which was true until the first tool that reads a database arrived and
        legitimately needed one. A tool that is neither a source nor a sink has nothing outside the
        frames it was handed, so its examples must still run against a bare process.
        """
        example = WorkedExample(
            scenario="Label every row",
            configuration={"label": "x"},
            explanation="Every row gains a label column holding 'x'.",
            inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
            outputs=Ports({"out": pl.DataFrame({"id": [1], "label": ["x"]})}),
        )

        assert example.setup is None, "an example declares a world only when it needs one"
        check_tool(_StubTool(examples=(example,)))

        for tool in REGISTRY.tools():
            if tool.category in {"input", "output"}:
                continue
            for shipped in tool.help().examples:
                assert shipped.setup is None, (
                    f"{tool.name} is neither a source nor a sink, so its examples must need no "
                    f"prepared world, but one declares a setup"
                )

    def test_a_declared_world_is_not_a_configuration_field(self, process_left_as_found: None) -> None:
        """It travels on the example, so nothing that reads a configuration may ever see it.

        A tool whose own template happens to declare a field called ``setup`` is the case that
        catches an implementation routing the declaration through the configuration: the filled
        configuration must hold the template's fields and nothing else, and certainly not a
        callable. The template rules still bite through the new path too -- a nested default that
        could never be applied is refused on an example that declares a world exactly as on one
        that does not.
        """
        template = {
            "label": ConfigField(type="string", description="The label to add.", required=True),
            "setup": ConfigField(type="string", description="A field that shares the name, and nothing else."),
        }
        events: list[str] = []
        seen, execute_fn = _recorder()
        example = WorkedExample(
            scenario="A template field of the same name",
            configuration={"label": "x"},
            explanation="The tool's own 'setup' field is a string, unrelated to the example's declaration.",
            inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
            outputs=Ports({"out": pl.DataFrame({"id": [1]})}),
            setup=_prepared_world(events),
        )

        check_tool(_StubTool(template=template, execute_fn=execute_fn, examples=(example,)))

        assert events == ["enter", "exit"]
        assert seen == [{"label": "x", "setup": None}], "a filled configuration is the template, and no more"
        assert apply_defaults(template, {"label": "x"}) == {"label": "x", "setup": None}

        nested = dict(template)
        nested["rules"] = ConfigField(
            type="array",
            description="Ordered rules.",
            items=ConfigField(type="object", description="One rule.", default={}),
        )
        with pytest.raises(ConformanceError) as caught:
            check_tool(_StubTool(template=nested, execute_fn=execute_fn, examples=(example,)))

        assert "never applied" in str(caught.value)


class TestTemporaryEnvironmentPutsBackWhatWasThere:
    """The helper a source's or a sink's setup uses, so no tool hand-rolls save-and-restore."""

    def test_a_variable_that_already_had_a_value_comes_back_with_it(self, process_left_as_found: None) -> None:
        """Restored, not merely unset -- and restored when the block fails as well as when it does not.

        A catalog check runs inside somebody else's process, so a variable that already held a real
        connection must still hold it afterwards: unsetting it would break the run that asked for
        the check, which is worse than not checking. A variable that was absent goes back to being
        absent, which is the half a save-and-restore written by hand gets wrong.
        """
        held = "ETL_TEST_EXAMPLE_WORLD_ALREADY_SET"
        os.environ[held] = "the value the process was using"
        os.environ.pop(_PREPARED, None)

        with temporary_environment(**{held: "the value the example needs", _PREPARED: _PREPARED_VALUE}):
            assert os.environ[held] == "the value the example needs"
            assert os.environ[_PREPARED] == _PREPARED_VALUE

        assert os.environ[held] == "the value the process was using"
        assert _PREPARED not in os.environ, "a variable that was absent must go back to being absent"

        with pytest.raises(RuntimeError), temporary_environment(**{held: "briefly"}):
            raise RuntimeError("an example that fell over")

        assert os.environ[held] == "the value the process was using"


# ---------------------------------------------------------------------------
# An example is given a directory to work in
#
# ``setup`` solved the litter: an example that needs a file on disk builds it on entry to a
# declared context manager and tears it down in a ``finally``, so nothing survives a check. It
# cannot solve legibility, because the path has to sit in the example's frozen ``configuration``
# before any context is entered, so every source and every sink had to invent an absolute path and
# publish it on the catalog page. A reader who copies ``path: /tmp/<something>.xlsx`` writes their
# own report into ``/tmp``, which is not an instruction anybody meant to give.
#
# So the checker owns a directory for the duration of one example. It is created before the
# declared setup is entered -- wrapping it, not nested inside it -- and removed after the example
# has been checked, whatever the example did. A ``{scratch}`` token anywhere in the configuration
# resolves against that directory while the example runs, and the *stored* configuration is left
# holding the token, which is the whole point: the page shows ``{scratch}/figures.xlsx`` and the
# run uses a real directory nobody has to clean up.
#
# ``{scratch}`` and ``${...}`` are one character apart and mean opposite things: the first is a
# thing this module resolves, the second is a pipeline parameter somebody else was supposed to
# resolve and did not, which is still refused. ``${scratch}`` is therefore the trap worth pinning
# -- it contains the token's own spelling, and an implementation that substitutes text before
# looking for the placeholder would rescue exactly the configuration that should be refused.
#
# What this does not buy: a file written *outside* the directory is as invisible as it ever was.
# The environment and the working directory are still the only process state compared, and nothing
# here watches the filesystem at large. What it buys is that a correct example no longer needs to
# write outside, that anything left inside goes away whatever happened, and that a world which
# destroys the directory the checker owns is refused by name instead of breaking the next example.
# ---------------------------------------------------------------------------


# The token as a reader of the catalog page sees it, and what a prepared file says when read back.
_SCRATCH_FILE = "{scratch}/rows.csv"
_SCRATCH_CONTENT = b"first\nsecond\n"


def _scratch_path(value: str) -> Path:
    """The library's resolver, reached through the module rather than imported by name.

    Only the tests in this section need it, and reaching it through the module keeps a missing
    resolver failing those tests instead of failing collection for the whole file.
    """
    return tool_contract.scratch_path(value)


def _tolerated_scratch_path(value: str) -> Path | None:
    """The resolver where a caller has no right to expect a directory, as when the example is refused."""
    try:
        return _scratch_path(value)
    except ToolError:
        return None


def _scratch_world(
    events: list[str],
    opened: list[Path],
    *,
    contents: list[list[str]] | None = None,
    destroys_the_directory: bool = False,
) -> Callable[[], AbstractContextManager[None]]:
    """A setup that prepares its file inside the directory the checker opened, and nowhere else.

    ``destroys_the_directory`` is the mistake: a teardown that reaches past its own file and takes
    the checker's directory with it.
    """

    @contextmanager
    def prepare() -> Iterator[None]:
        target = _scratch_path(_SCRATCH_FILE)
        events.append("enter")
        opened.append(target)
        if contents is not None:
            contents.append(sorted(entry.name for entry in target.parent.iterdir()))
        target.write_bytes(_SCRATCH_CONTENT)
        try:
            yield
        finally:
            events.append("exit")
            if destroys_the_directory:
                shutil.rmtree(target.parent, ignore_errors=True)

    return prepare


def _recording_scratch_world(opened: list[Path | None]) -> Callable[[], AbstractContextManager[None]]:
    """A setup that only notes whichever directory it was given, if it was given one at all."""

    @contextmanager
    def prepare() -> Iterator[None]:
        opened.append(_tolerated_scratch_path("{scratch}"))
        yield

    return prepare


class _ScratchReadingTool(_StubTool):
    """A tool whose every phase needs the file its example's token names.

    Each phase refuses a configuration still holding the token and refuses a path whose directory
    is not there, so the recorded events are a claim about coverage: a directory opened around the
    run alone would leave ``validate_config`` -- where a source resolves what it is about to read
    -- looking at a token and an absent directory.
    """

    def __init__(
        self,
        events: list[str],
        setup: Callable[[], AbstractContextManager[None]] | None,
        *,
        documented: str = "first",
        raises: bool = False,
    ) -> None:
        super().__init__(
            name="scratch_reading",
            template={"path": ConfigField(type="string", description="Where the rows are.", required=True)},
            examples=(
                WorkedExample(
                    scenario="Read the file the example prepared",
                    configuration={"path": _SCRATCH_FILE},
                    explanation="The label column holds the first word of the file at 'path'.",
                    inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
                    outputs=Ports({"out": pl.DataFrame({"id": [1], "label": [documented]})}),
                    setup=setup,
                ),
            ),
        )
        self.events = events
        self.configs: list[Mapping[str, Any]] = []
        self._raises = raises

    def _phase(self, phase: str, config: Mapping[str, Any]) -> None:
        self.events.append(phase)
        self.configs.append(dict(config))
        path = str(config["path"])
        if "{scratch}" in path:
            raise ConfigError(f"{phase} was handed the token {path!r} rather than a directory to work in")
        if not Path(path).parent.is_dir():
            raise ConfigError(f"{phase} ran before the directory the token resolved to existed")

    def validate_config(self, config: Mapping[str, Any]) -> None:
        self._phase("validate_config", config)
        super().validate_config(config)

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        self._phase("validate_inputs", config)

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self._phase("execute", config)
        if self._raises:
            raise RuntimeError("the engine fell over")
        first = Path(str(config["path"])).read_text().split()[0]
        return Ports({"out": inputs["in"].with_columns(pl.lit(first).alias("label"))})


def _nested_scratch_template() -> dict[str, ConfigField]:
    """A path at the top level, one inside an object, and one inside an array of objects.

    The last shape is the one a reader of the catalog most often meets: the CSV reader's list of
    columns to correct is an array of objects, so a resolver that walked only the top level would
    leave a token sitting in an element nobody looks at twice.
    """
    return {
        "path": ConfigField(type="string", description="Where the rows go.", required=True),
        "options": ConfigField(
            type="object",
            description="Where anything else goes.",
            fields={"folder": ConfigField(type="string", description="The directory to work in.")},
        ),
        "column_types": ConfigField(
            type="array",
            description="One entry per column to correct.",
            items=ConfigField(
                type="object",
                description="One column and where its correction lives.",
                fields={
                    "column": ConfigField(type="string", description="The column.", required=True),
                    "at": ConfigField(type="string", description="Where its correction lives.", required=True),
                },
            ),
        ),
    }


def _nested_scratch_config() -> dict[str, Any]:
    """A token at every depth a value can sit at, beside a plain relative path that is not one."""
    return {
        "path": "{scratch}/rows.csv",
        "options": {"folder": "{scratch}"},
        "column_types": [
            {"column": "key", "at": "{scratch}/key.csv"},
            {"column": "label", "at": "beside.csv"},
        ],
    }


def _scratch_example(configuration: Mapping[str, Any], setup: Any = None) -> WorkedExample:
    """One pass-through example over whichever configuration a case wants checked."""
    return WorkedExample(
        scenario="Work inside the directory the checker opened",
        configuration=configuration,
        explanation="The rows arrive unchanged; where the work happens is the interesting part.",
        inputs=Ports({"in": pl.DataFrame({"id": [1]})}),
        outputs=Ports({"out": pl.DataFrame({"id": [1]})}),
        setup=setup,
    )


def _writes_a_file(inputs: Ports, config: Mapping[str, Any]) -> Ports:
    """A sink's behaviour in miniature: leave a file behind at the configured path."""
    Path(str(config["path"])).write_bytes(_SCRATCH_CONTENT)
    return Ports({"out": inputs["in"]})


def _unresolved_parameter_cases() -> list[Any]:
    """(configuration, the text a refusal must quote) for each way a parameter arrives unresolved."""
    return [
        pytest.param({"path": "${target}"}, "${target}", id="a_parameter_on_its_own"),
        pytest.param({"path": "${scratch}"}, "${scratch}", id="a_parameter_spelled_like_the_token"),
        pytest.param(
            {"path": "{scratch}/rows.csv", "options": {"folder": "${target}"}},
            "${target}",
            id="a_parameter_beside_a_token",
        ),
        pytest.param(
            {
                "path": "{scratch}/rows.csv",
                "column_types": [{"column": "key", "at": "${target}/key.csv"}],
            },
            "${target}",
            id="a_parameter_beside_a_token_inside_an_element",
        ),
    ]


class TestAnExampleIsGivenADirectoryToWorkIn:
    """The token on the page, the directory underneath it, and nothing left over either way."""

    def test_the_token_is_a_real_directory_for_the_setup_and_for_every_phase(
        self,
        process_left_as_found: None,
    ) -> None:
        """Created before the setup, resolved for every phase, removed after, token still on the page.

        The directory wraps the declared setup rather than sitting inside it, because the setup is
        what prepares the file the example reads and it has nowhere to put it otherwise. It has to
        outlast validation too: a source resolves what it is about to read in ``validate_config``,
        and a directory opened around the run alone would hand that phase a token.

        The stored configuration is the deliverable. Checking an example must not rewrite it, or
        the catalog page shows a directory that existed for a few milliseconds on somebody else's
        machine instead of the token a reader can copy.
        """
        events: list[str] = []
        opened: list[Path] = []
        tool = _ScratchReadingTool(events, _scratch_world(events, opened))
        stored = deepcopy(tool.help().examples[0].configuration)

        with pytest.raises(ToolError):
            _scratch_path(_SCRATCH_FILE)

        check_tool(tool)

        assert list(dict.fromkeys(events)) == ["enter", "validate_config", "validate_inputs", "execute", "exit"]
        assert len(opened) == 1, f"one directory for one example: {opened}"
        prepared = opened[0]
        assert prepared.is_absolute(), f"a token must resolve to somewhere real: {prepared}"
        # Compared as paths, not text: on Windows the token resolves to "...\scratch_x/rows.csv", which
        # names the same file as "...\scratch_x\rows.csv" but is not the same string.
        resolved = {Path(config["path"]) for config in tool.configs}
        assert resolved == {prepared}, f"every phase saw the same resolved path: {resolved}"
        assert tool.help().examples[0].configuration == stored == {"path": _SCRATCH_FILE}
        assert not prepared.exists(), "the file the example prepared must not outlive it"
        assert not prepared.parent.exists(), "nor the directory it was prepared in"
        with pytest.raises(ToolError):
            _scratch_path(_SCRATCH_FILE)

    @pytest.mark.parametrize(
        ("documented", "raises", "refusal"),
        [
            pytest.param("something else", False, "drifted", id="the_example_drifts"),
            pytest.param("first", True, "does not run", id="the_tool_raises"),
        ],
    )
    def test_the_directory_is_removed_even_when_the_example_fails_or_the_tool_raises(
        self,
        documented: str,
        raises: bool,
        refusal: str,
        process_left_as_found: None,
    ) -> None:
        """A red check must leave no more behind than a green one, and must still say what was red.

        The two ways a check ends badly are a comparison that fails and a tool that throws. A
        directory removed only on the happy path would turn every failed check into litter, and a
        removal that swallowed the failure would turn a drifted example into a silent pass.
        """
        events: list[str] = []
        opened: list[Path] = []
        tool = _ScratchReadingTool(events, _scratch_world(events, opened), documented=documented, raises=raises)

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        assert refusal in str(caught.value), f"the original failure must still be the one reported: {caught.value}"
        assert events[0] == "enter" and events[-1] == "exit", f"prepared first, torn down last: {events}"
        assert opened and not opened[0].parent.exists(), f"the directory outlived a failed check: {opened}"

    def test_the_token_resolves_at_every_depth_and_the_stored_configuration_keeps_it(self) -> None:
        """Top level, inside an object, inside an element of an array -- and nothing rewritten.

        The stored configuration is compared whole rather than field by field, because the
        cheapest way to resolve a nested value is to rewrite the dictionary that holds it, which
        would leave the catalog page showing a resolved path for the nested field and the token for
        the others. A plain relative path sitting beside the tokens is left exactly as written.
        """
        seen, execute_fn = _recorder()
        configuration = _nested_scratch_config()
        stored = deepcopy(configuration)
        tool = _StubTool(
            template=_nested_scratch_template(),
            execute_fn=execute_fn,
            examples=(_scratch_example(configuration),),
        )

        check_tool(tool)

        assert len(seen) == 1, f"one run of one example: {seen}"
        used = seen[0]
        directory = Path(str(used["options"]["folder"]))
        assert directory.is_absolute(), f"the token must resolve to a directory: {used}"
        assert Path(str(used["path"])) == directory / "rows.csv"
        assert Path(str(used["column_types"][0]["at"])) == directory / "key.csv"
        assert used["column_types"][1]["at"] == "beside.csv", "a path that holds no token is not a path to rewrite"
        assert configuration == stored, "checking an example must not rewrite what the page renders"
        assert not directory.exists(), f"the directory outlived the example: {directory}"

    @pytest.mark.parametrize(("configuration", "quoted"), _unresolved_parameter_cases())
    def test_an_unresolved_parameter_is_still_refused_beside_a_token_and_when_spelled_like_one(
        self,
        configuration: Mapping[str, Any],
        quoted: str,
        process_left_as_found: None,
    ) -> None:
        """``${...}`` is somebody else's unfinished work, and resolving text must not rescue it.

        ``${scratch}`` is the case that decides the order of the two rules: it *contains* the
        token's own spelling, so an implementation that substitutes first and looks for the
        placeholder afterwards would turn a configuration nobody resolved into one that runs,
        against a directory that disappears the moment the check ends.

        A configuration mixing the two is refused for the parameter and not excused by the token,
        the refusal quotes the text that has to change, and the example never reaches the engine --
        a run failure would point at the tool, and the tool is not what is wrong here.
        """
        seen, execute_fn = _recorder()
        opened: list[Path | None] = []
        tool = _StubTool(
            template=_nested_scratch_template(),
            execute_fn=execute_fn,
            examples=(_scratch_example(configuration, setup=_recording_scratch_world(opened)),),
        )

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        message = str(caught.value)
        assert "unresolved" in message, f"a refusal must say the parameter was never resolved: {message}"
        assert quoted in message, f"a refusal must quote the text to change: {message}"
        assert seen == [], "an example that cannot be valid must never reach the engine"
        assert all(directory is None or not directory.exists() for directory in opened), (
            f"a refused example left a directory behind: {opened}"
        )

    def test_an_example_that_uses_no_token_is_checked_exactly_as_before(self) -> None:
        """The fence: thirteen tools have no world and no token, and nothing may change for them.

        A configuration that merely contains braces is the other half of the fence. ``{scratch}``
        is one token with one meaning, not the start of a substitution language, so a label reading
        ``{total}`` reaches the tool as written -- an implementation reaching for ``str.format``
        would either rewrite it or fall over on it, and both are changes to tools that asked for
        nothing.
        """
        seen, execute_fn = _recorder()
        opened: list[Path | None] = []
        tool = _StubTool(
            execute_fn=execute_fn,
            examples=(_scratch_example({"label": "x"}, setup=_recording_scratch_world(opened)),),
        )

        check_tool(tool)

        assert seen == [{"label": "x"}], f"a token-free configuration is handed over as written: {seen}"
        assert opened == [None], f"an example with no token to resolve is given no directory: {opened}"

        braced = _StubTool(execute_fn=execute_fn, examples=(_scratch_example({"label": "{total}"}),))

        check_tool(braced)

        assert seen[-1] == {"label": "{total}"}, f"an unknown brace is ordinary text: {seen[-1]}"

    def test_a_world_that_destroys_the_directory_is_refused_and_one_that_fills_it_is_not(
        self,
        process_left_as_found: None,
    ) -> None:
        """The one leak the contract can now see, and the one it must not mistake for a leak.

        A sink's example is *supposed* to leave a file behind: writing one is what it documents.
        Anything inside the directory goes with it, so that is no refusal at all. A teardown that
        removes the directory itself is the mistake -- it is the checker's, not the example's, and
        an implementation that shrugged at it would carry on into the next example with nothing to
        work in. A file written somewhere else entirely remains as invisible as it always was.
        """
        events: list[str] = []
        opened: list[Path] = []
        tool = _ScratchReadingTool(events, _scratch_world(events, opened, destroys_the_directory=True))

        with pytest.raises(ConformanceError) as caught:
            check_tool(tool)

        assert opened, "the setup was never given a directory to destroy"
        message = str(caught.value)
        assert tool.name in message, f"a refusal must name the tool it came from: {message}"
        assert "Read the file the example prepared" in message, f"and the example it means: {message}"
        assert str(opened[0].parent) in message, f"and the directory that went missing: {message}"

        left_behind: list[Mapping[str, Any]] = []

        def execute_fn(inputs: Ports, config: Mapping[str, Any]) -> Ports:
            left_behind.append(dict(config))
            return _writes_a_file(inputs, config)

        sink = _StubTool(
            template=_nested_scratch_template(),
            execute_fn=execute_fn,
            examples=(_scratch_example({"path": _SCRATCH_FILE}),),
        )

        check_tool(sink)

        written = Path(str(left_behind[0]["path"]))
        assert not written.exists(), f"a file left inside the directory must go with it: {written}"
        assert not written.parent.exists(), f"and so must the directory: {written.parent}"

    def test_every_check_of_the_same_example_gets_a_directory_of_its_own(
        self,
        process_left_as_found: None,
    ) -> None:
        """A catalog is checked more than once a process, and a sink's example says so out loud.

        Its second check must find the directory as empty as its first did: a directory shared
        between checks would let a file written by one run satisfy the next, so an example claiming
        to write a workbook where nothing exists yet would pass without ever writing one.
        """
        events: list[str] = []
        opened: list[Path] = []
        contents: list[list[str]] = []
        registry = ToolRegistry()
        registry.register(_ScratchReadingTool(events, _scratch_world(events, opened, contents=contents)))

        check_registry(registry)
        check_registry(registry)

        assert events.count("enter") == 2 and events.count("exit") == 2, f"once per check: {events}"
        assert contents == [[], []], f"each check began with an empty directory: {contents}"
        assert opened[0].parent != opened[1].parent, f"the second check reused the first's directory: {opened}"
        assert not any(target.parent.exists() for target in opened), f"a directory outlived its check: {opened}"
