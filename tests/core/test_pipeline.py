"""Tests for the pipeline specification and its loader: a flow as a data file, not as code.

Nineteen tools exist, each with a typed configuration template, declared input and output ports and
a boundary that validates everything before a row moves. What did not exist is a way to say *which*
tools, wired *how* -- and that artefact is the point: a flow a person can read and change, where
changing it changes what actually runs. A flow written as functions can only be edited by generating
functions; a flow written as data is edited by editing it.

So this module specifies the data model and its loader, and nothing that moves a row. Everything
below is a refusal that can be reached before execution begins, plus the two things a loader must
hand a runner: the order to walk the steps in, and the column references a runner will have to check
once it has frames.

**An edge is port-to-port.** ``kept.true`` names a step and one of that step's *declared output
ports*, never merely a step. That is what makes the rows a filter discards a thing a flow can route
-- to a report, to an assertion, to nothing -- rather than a number a reader reconstructs afterwards
with an anti-join. Both halves of an edge are checked here: the step must exist, and the port must be
one its tool declares.

**Parameters are resolved at load.** A configuration may say ``'${rules.kept_code}'``, and the
loader substitutes it from a rules mapping before the tool's own template ever sees the value. This
matters more than it looks. The tool boundary refuses an unresolved ``${...}`` precisely because an
unresolved placeholder *compiles perfectly well*, matches nothing, and leaves the run reporting
success -- so resolution has to happen before validation, a rule name nobody defined has to be
refused by name, and a resolved configuration has to carry no ``${`` at any nesting depth. The
``{scratch}`` token is a different thing that belongs to the catalog checker, and must survive
untouched.

**Load-time validation is structural, and says so.** ``references("[code] = 'A'")`` yields
``{"code"}``, but whether a column called ``code`` exists needs the upstream frame, and a source's
columns are not knowable without connecting to it. So what is checked here is that every expression
*compiles as syntax* -- which ``validate_against_template`` already does -- and the column references
are published for whoever holds the frames. Claiming more would be a check that passes and a run that
fails.

Vocabulary here is deliberately neutral -- entries, headers, codes, keys -- because the library ships
into other repositories and must carry no caller's nouns.
"""

import os
import re
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl
import pytest

import dantron_etl
from dantron_etl.expr import compile_expr
from dantron_etl.pipeline import (
    Lane,
    PipelineError,
    PipelineSpec,
    PortRef,
    StepSpec,
    load_pipeline,
    read_pipeline_file,
)
from dantron_etl.runner import run_pipeline
from dantron_etl.tool import REGISTRY, ConfigError, ToolError, apply_defaults, validate_against_template

# The environment a child interpreter is started with, when a test needs one with nothing inherited.
# Linux needs only a PATH. Windows will not start Python without SYSTEMROOT, so it is passed on
# where it exists.
_BARE_ENVIRONMENT: Final[dict[str, str]] = {
    "PATH": os.pathsep.join(["/usr/bin", "/bin"]) if os.name != "nt" else os.environ.get("PATH", ""),
    **({"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {}),
}

# ---------------------------------------------------------------------------
# The fixture flow
#
# Two sources, a split, a pairing and a write. Every tool it names is one the shipped registry
# already holds, so the loader is exercised against real templates and real port declarations rather
# than against a stub that agrees with it.
# ---------------------------------------------------------------------------

#: The rules a pipeline's configurations may reference. A flat name-to-value mapping on purpose: a
#: rules *file* in this repository states a reason beside every value, and that shape is the
#: consumer's business. The library is handed the values and nothing else.
RULES: Final[Mapping[str, Any]] = {
    "kept_code": "A",
    "pair_keys": ["entry_id"],
    "key_column": "entry_id",
}


def _document() -> dict[str, Any]:
    """A fresh, valid pipeline document. Every test below mutates a copy of this one."""
    return {
        "name": "a neutral fixture",
        "lanes": [
            {"id": "intake", "label": "Intake", "order": 1},
            {"id": "shaping", "label": "Shaping", "order": 2},
        ],
        "steps": [
            {
                "id": "entries",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "{scratch}/entries.csv"},
            },
            {
                "id": "headers",
                "tool": "read_csv",
                "lane": "intake",
                "stage": "collect",
                "config": {"path": "{scratch}/headers.csv"},
            },
            {
                "id": "kept",
                "tool": "filter",
                "lane": "shaping",
                "stage": "shape",
                "why": "only the codes the rules admit travel on",
                "config": {"predicate": "[code] = '${rules.kept_code}'"},
                "inputs": {"rows": "entries.rows"},
            },
            {
                "id": "paired",
                "tool": "join",
                "lane": "shaping",
                "stage": "shape",
                "config": {"keys": "${rules.pair_keys}"},
                "inputs": {"left": "kept.true", "right": "headers.rows"},
            },
            {
                "id": "published",
                "tool": "write_csv",
                "lane": "shaping",
                "stage": "publish",
                "config": {"path": "{scratch}/paired.csv"},
                "inputs": {"rows": "paired.joined"},
            },
        ],
    }


def _step(document: Mapping[str, Any], step_id: str) -> dict[str, Any]:
    """The mutable step dict with ``step_id``, so a test can express its mutation in one line."""
    for step in document["steps"]:
        if step["id"] == step_id:
            return step
    raise AssertionError(f"the fixture document has no step {step_id!r}")


def _filter_step(step_id: str, source: str) -> dict[str, Any]:
    """A neutral extra split, reading ``source`` -- used to build fan-out, dead ends and cycles."""
    return {
        "id": step_id,
        "tool": "filter",
        "lane": "shaping",
        "stage": "shape",
        "config": {"predicate": "[code] = 'A'"},
        "inputs": {"rows": source},
    }


def _writer_step(step_id: str, source: str) -> dict[str, Any]:
    """A neutral sink, so a branch under test is not also a branch that goes nowhere."""
    return {
        "id": step_id,
        "tool": "write_csv",
        "lane": "shaping",
        "stage": "publish",
        "config": {"path": f"{{scratch}}/{step_id}.csv"},
        "inputs": {"rows": source},
    }


def _placeholders(value: Any) -> list[str]:
    """Every string anywhere inside ``value`` that still holds ``${``, at any nesting depth."""
    if isinstance(value, str):
        return [value] if "${" in value else []
    if isinstance(value, (list, tuple)):
        return [found for item in value for found in _placeholders(item)]
    if isinstance(value, Mapping):
        return [found for item in value.values() for found in _placeholders(item)]
    return []


class TestAPipelineIsADataFile:
    """The shape a loaded pipeline has, and the two answers it owes whoever runs it."""

    def test_a_document_loads_into_steps_edges_and_an_order(self) -> None:
        """The whole surface: steps as authored, edges port-to-port, and a walkable order."""
        spec = load_pipeline(_document(), rules=RULES)

        assert isinstance(spec, PipelineSpec)
        assert spec.name == "a neutral fixture"
        assert tuple(step.id for step in spec.steps) == ("entries", "headers", "kept", "paired", "published")

        kept = spec.step("kept")
        assert kept.tool == "filter"
        assert kept.why == "only the codes the rules admit travel on"
        # As authored, with parameters resolved and nothing else added. The file a person edits and
        # the object a canvas draws must be the same thing: filling 'nulls' in here would show a
        # reader a configuration nobody wrote. run_tool fills defaults at the moment of running.
        assert dict(kept.config) == {"predicate": "[code] = 'A'"}
        assert dict(kept.inputs) == {"rows": PortRef("entries", "rows")}

        # A source declares no inputs and carries no explanation it was not given.
        assert dict(spec.step("entries").inputs) == {}
        assert spec.step("entries").why is None

        # An edge names a step *and one of its output ports*, which is what lets the rows a filter
        # discarded be routed rather than reconstructed.
        assert dict(spec.step("paired").inputs) == {
            "left": PortRef("kept", "true"),
            "right": PortRef("headers", "rows"),
        }
        assert spec.consumers(PortRef("kept", "true")) == (PortRef("paired", "left"),)
        assert spec.consumers(PortRef("kept", "false")) == ()

        # Layout and scheduling travel in the same artefact as the wiring, so a canvas cannot drift
        # from the flow it draws.
        assert spec.lanes == (Lane("intake", "Intake", 1), Lane("shaping", "Shaping", 2))
        assert kept.lane == "shaping"
        assert spec.stages() == ("collect", "shape", "publish")

        # The order a runner walks: dependency order, ties broken by the order the author wrote.
        assert spec.order() == ("entries", "headers", "kept", "paired", "published")

        shuffled = _document()
        shuffled["steps"] = list(reversed(shuffled["steps"]))
        assert load_pipeline(shuffled, rules=RULES).order() == (
            "headers",
            "entries",
            "kept",
            "paired",
            "published",
        )

    def test_an_unregistered_tool_is_refused_with_the_catalog_beside_it(self) -> None:
        """A tool nobody registered names itself, its step, and what the registry does hold."""
        assert issubclass(PipelineError, ToolError), "a loader refusal must stay inside the one error family"

        document = _document()
        _step(document, "kept")["tool"] = "fliter"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "fliter" in message
        assert "kept" in message
        assert "filter" in message and "join" in message, "the refusal must list what is registered"

    def test_a_configuration_its_tool_refuses_is_refused_at_load(self) -> None:
        """Both halves a tool can refuse without frames, delegated rather than re-implemented."""
        document = _document()
        _step(document, "kept")["config"]["nulls"] = "maybe"

        with pytest.raises(ConfigError) as raised:
            load_pipeline(document, rules=RULES)

        # Proof of delegation rather than a second implementation: the template's own words, verbatim.
        with pytest.raises(ConfigError) as delegated:
            validate_against_template(
                REGISTRY.get("filter").config_template(),
                {"predicate": "[code] = 'A'", "nulls": "maybe"},
            )
        assert str(delegated.value) in str(raised.value)
        assert "kept" in str(raised.value)

        # A cross-field rule needs no frames either, so a pairing with no key columns is a load-time
        # refusal too -- a tool's own validate_config, run on the filled configuration.
        keyless = _document()
        _step(keyless, "paired")["config"] = {}
        with pytest.raises(ConfigError) as no_keys:
            load_pipeline(keyless, rules=RULES)
        join = REGISTRY.get("join")
        with pytest.raises(ConfigError) as delegated_rule:
            join.validate_config(apply_defaults(join.config_template(), {}))
        assert str(delegated_rule.value) in str(no_keys.value)
        assert "paired" in str(no_keys.value)


class TestTheWiringIsRefusedBeforeARowMoves:
    """Every way an edge can be wrong, each named where it is and each with its own refusal."""

    def test_an_input_port_the_tool_does_not_declare_is_refused(self) -> None:
        document = _document()
        _step(document, "kept")["inputs"] = {"lines": "entries.rows"}

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "lines" in message, "a port name the tool never heard of is the likelier mistake, so it is reported"
        assert "kept" in message
        assert "rows" in message, "the refusal must list the input ports the tool does declare"

    def test_a_declared_input_port_left_unfed_is_refused(self) -> None:
        """Half a pairing is not a pairing; nothing would ever arrive on the other side."""
        document = _document()
        _step(document, "paired")["inputs"] = {"left": "kept.true"}

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "right" in message
        assert "paired" in message
        assert "join" in message

    def test_an_output_port_nobody_declares_cannot_be_the_far_end_of_an_edge(self) -> None:
        document = _document()
        _step(document, "paired")["inputs"]["left"] = "kept.matched"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "matched" in message
        assert "kept" in message, "the refusal must name the step that was supposed to produce it"
        assert "paired" in message, "and the step the offending edge sits in"
        assert "true" in message and "false" in message, "and what that step does declare"

        # An edge with no port at all is the same mistake earlier: a step is not a port.
        portless = _document()
        _step(portless, "kept")["inputs"]["rows"] = "entries"
        with pytest.raises(PipelineError) as bare:
            load_pipeline(portless, rules=RULES)
        assert "entries" in str(bare.value)
        assert "kept" in str(bare.value)
        assert "port" in str(bare.value).lower()

    def test_an_edge_naming_a_step_that_does_not_exist_is_refused(self) -> None:
        document = _document()
        _step(document, "paired")["inputs"]["right"] = "hedaers.rows"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "hedaers" in message
        assert "paired" in message
        assert "headers" in message, "the refusal must list the steps there are"

    def test_a_duplicate_step_id_is_refused(self) -> None:
        """Ids are how every edge names its far end, so two steps cannot share one."""
        document = _document()
        document["steps"].append(_filter_step("kept", "headers.rows"))

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "kept" in message
        assert "duplicate" in message.lower()

    def test_a_cycle_is_refused_naming_the_steps_in_it(self) -> None:
        """A flow is walked from its sources, so a loop has nowhere to start -- and which loop matters."""
        document = _document()
        document["steps"].append(_filter_step("loop_a", "loop_b.true"))
        document["steps"].append(_filter_step("loop_b", "loop_a.true"))

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "cycle" in message.lower()
        assert "loop_a" in message and "loop_b" in message
        assert "entries" not in message, "the steps that are fine must not be named in the refusal"

    def test_an_output_port_may_feed_two_steps(self) -> None:
        """Fan-out is legal: one port, two consumers, each getting the frame as it was produced.

        The envelope already supports it -- Ports is read-only and run_tool refuses a tool that
        writes into its own input -- so refusing it here would forbid the ordinary shape of routing
        one set of rows to two different ends for no gain. What fan-out is not is fan-*in*: an input
        port takes exactly one edge, which the document's shape already enforces by making inputs a
        mapping keyed by port name.
        """
        document = _document()
        document["steps"].append(_filter_step("also_kept", "entries.rows"))
        document["steps"].append(_writer_step("also_published", "also_kept.true"))

        spec = load_pipeline(document, rules=RULES)

        assert spec.consumers(PortRef("entries", "rows")) == (
            PortRef("kept", "rows"),
            PortRef("also_kept", "rows"),
        )

    def test_a_step_may_read_two_ports_of_one_upstream_step(self, tmp_path: Path) -> None:
        """Two edges from one step into one step, on different ports, are not a loop.

        A join reading both halves of one split is the ordinary way to set what passed against what
        did not. The loader once counted the second edge as a second dependency it then cleared
        twice, put the join in its walk twice, and refused the flow as a cycle naming no steps.
        """
        entries = tmp_path / "entries.csv"
        entries.write_text("group,code,amount\ng1,A,10\ng1,B,5\ng2,A,7\n", encoding="utf8")
        paired = tmp_path / "paired.csv"
        document = {
            "name": "a split read twice",
            "steps": [
                {"id": "entries", "tool": "read_csv", "config": {"path": str(entries)}},
                {
                    "id": "split",
                    "tool": "filter",
                    "config": {"predicate": "[code] = 'A'"},
                    "inputs": {"rows": "entries.rows"},
                },
                {
                    "id": "paired",
                    "tool": "join",
                    "config": {"keys": ["group"]},
                    "inputs": {"left": "split.true", "right": "split.false"},
                },
                {
                    "id": "published",
                    "tool": "write_csv",
                    "config": {"path": str(paired)},
                    "inputs": {"rows": "paired.joined"},
                },
            ],
        }

        spec = load_pipeline(document)

        assert spec.order() == ("entries", "split", "paired", "published"), "each step walked once"
        run_pipeline(spec)
        written = pl.read_csv(paired)
        assert written["group"].to_list() == ["g1"]
        assert written["amount"].to_list() == [10]
        assert written["amount_right"].to_list() == [5]

    def test_a_cycle_through_two_ports_of_one_step_is_still_refused_by_name(self) -> None:
        """The mirror of the case above: parallel edges inside a real loop must not hide the loop."""
        document = _document()
        document["steps"].append(
            {
                "id": "loop_a",
                "tool": "join",
                "lane": "shaping",
                "stage": "shape",
                "config": {"keys": ["entry_id"]},
                "inputs": {"left": "loop_b.true", "right": "loop_b.false"},
            }
        )
        document["steps"].append(_filter_step("loop_b", "loop_a.joined"))

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "cycle" in message.lower()
        assert "loop_a" in message and "loop_b" in message
        assert "entries" not in message, "the steps that are fine must not be named in the refusal"

    def test_a_step_nothing_consumes_and_that_writes_nothing_is_refused(self) -> None:
        """A branch that goes nowhere is refused; a branch that ends in a sink is the normal case.

        A dead transform computes rows nobody reads and reports success, which in a log is
        indistinguishable from a branch that mattered -- and the likeliest cause is an edge somebody
        meant to draw and did not. It is refusable here, by name, so it is refused here.

        The exemption is not a flag an author sets but the category its tool already publishes: an
        'output' or 'assert' tool has an effect that is not one of its ports, so nothing consuming it
        says nothing. That cannot be lied about the way a per-step 'terminal: true' could be.
        """
        # The base flow ends in a write_csv nobody consumes, and that is fine.
        load_pipeline(_document(), rules=RULES)

        document = _document()
        document["steps"].append(_filter_step("dropped", "entries.rows"))

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "dropped" in message
        assert "consume" in message.lower()
        assert "published" not in message, "the sink at the end of the flow is not a dead step"

    def test_a_step_in_a_lane_nobody_declared_is_refused(self) -> None:
        """Lanes live in the same file as the wiring, so a lane that is not there is a real defect."""
        document = _document()
        _step(document, "kept")["lane"] = "shapign"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "shapign" in message
        assert "kept" in message
        assert "shaping" in message, "the refusal must list the lanes there are"


class TestParametersAreResolvedAtLoad:
    """``${rules.*}`` is substituted before any tool sees it, or refused by name."""

    def test_a_rule_reference_is_resolved_before_the_configuration_is_validated(self) -> None:
        """Resolved everywhere, at every depth, with the value's own type -- and {scratch} untouched."""
        document = _document()
        _step(document, "entries")["config"]["column_types"] = [
            {"column": "${rules.key_column}", "type": "String"},
        ]

        spec = load_pipeline(document, rules=RULES)

        # Inside a larger string, the value is interpolated as text.
        assert spec.step("kept").config["predicate"] == "[code] = 'A'"

        # A reference that is the whole string takes the rule's own type: a list stays a list, which
        # is what a key-columns field is declared to take.
        assert spec.step("paired").config["keys"] == ["entry_id"]

        # And a reference nested inside an array of objects resolves too, because that is the shape
        # a real configuration has, not the exotic one.
        assert spec.step("entries").config["column_types"] == [{"column": "entry_id", "type": "String"}]

        # {scratch} belongs to the catalog checker and is not a parameter. It survives exactly.
        assert spec.step("entries").config["path"] == "{scratch}/entries.csv"

        # Nothing anywhere still holds a placeholder: an unresolved one compiles, matches nothing and
        # reports success, which is the whole reason the tool boundary refuses it.
        left = [found for step in spec.steps for found in _placeholders(step.config)]
        assert left == []

    def test_a_list_rule_inside_text_is_written_as_a_parenthesised_list(self) -> None:
        """So a list of allowed values can be used in an expression, which is the only place it fits.

        Square brackets would be read as a column reference, so the obvious rendering of a Python
        list produces an expression that names a column nobody has.
        """
        document = _document()
        _step(document, "kept")["config"]["predicate"] = "[code] in ${rules.kept_codes}"

        spec = load_pipeline(document, rules={**RULES, "kept_codes": ["A", "B"]})

        assert spec.step("kept").config["predicate"] == "[code] in ('A', 'B')"

    def test_a_reference_to_a_rule_that_does_not_exist_is_refused_at_load(self) -> None:
        document = _document()
        _step(document, "kept")["config"]["predicate"] = "[code] = '${rules.absent_code}'"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "absent_code" in message
        assert "kept" in message
        assert "kept_code" in message and "pair_keys" in message, "the refusal must list the rules on offer"

        # 'rules' is the only namespace there is; anything else is a placeholder nobody will ever
        # bind, and is refused here rather than reaching the tool boundary as a value that compiles.
        # The namespace is deliberately spelled over a rule that does exist, because a loader that
        # reads only the last segment would bind this one and never say so.
        stray = _document()
        _step(stray, "kept")["config"]["predicate"] = "[code] = '${elsewhere.kept_code}'"
        with pytest.raises(PipelineError) as unknown_namespace:
            load_pipeline(stray, rules=RULES)
        assert "elsewhere.kept_code" in str(unknown_namespace.value)
        assert "kept" in str(unknown_namespace.value)


class TestWhatTheLoaderCannotCheck:
    """The limit stated plainly, and the seam that hands the rest to whoever holds the frames."""

    def test_syntax_is_checked_at_load_and_columns_are_left_to_the_runner(self) -> None:
        """An expression that will not parse is refused now; one naming a column that may not exist
        is not, because a source's columns are not knowable without connecting to it. What the loader
        can do instead is publish the references, so the runner checks them the moment frames exist.
        """
        broken = _document()
        _step(broken, "kept")["config"]["predicate"] = "[code] = "
        with pytest.raises(ConfigError) as raised:
            load_pipeline(broken, rules=RULES)
        assert "kept" in str(raised.value)

        document = _document()
        _step(document, "kept")["config"]["predicate"] = "[nonexistent] = 'A'"
        spec = load_pipeline(document, rules=RULES)

        assert spec.column_references()["kept"]["predicate"] == frozenset({"nonexistent"})
        # Only expression fields: a join's key columns are plain strings, not an expression, so the
        # loader does not pretend to know they are columns of anything.
        assert "paired" not in spec.column_references()


class TestTheLibraryTakesNoFileFormatDependency:
    """The core loader takes a mapping; reading a file is a convenience behind a lazy import."""

    def test_the_core_takes_a_mapping_and_the_file_reader_is_a_lazy_extra(self, tmp_path: Path) -> None:
        """polars plus the standard library is the whole dependency set, and YAML is neither.

        This library ships into repositories that did not choose its dependencies, so the artefact it
        understands is a mapping and parsing a file is the consumer's business. The convenience for a
        consumer that wants it imports the parser only when called, and says what to install if it is
        not there -- the same pattern the database tools already use for a driver.
        """
        document = {
            "name": "a neutral fixture",
            "steps": [
                {"id": "entries", "tool": "read_csv", "config": {"path": "{scratch}/entries.csv"}},
                {
                    "id": "kept",
                    "tool": "filter",
                    "config": {"predicate": "[code] = '${rules.kept_code}'"},
                    "inputs": {"rows": "entries.rows"},
                },
                {
                    "id": "published",
                    "tool": "write_csv",
                    "config": {"path": "{scratch}/kept.csv"},
                    "inputs": {"rows": "kept.true"},
                },
            ],
        }
        written = tmp_path / "flow.yaml"
        written.write_text(
            "name: a neutral fixture\n"
            "steps:\n"
            "  - id: entries\n"
            "    tool: read_csv\n"
            '    config: {path: "{scratch}/entries.csv"}\n'
            "  - id: kept\n"
            "    tool: filter\n"
            "    config: {predicate: \"[code] = '${rules.kept_code}'\"}\n"
            "    inputs: {rows: entries.rows}\n"
            "  - id: published\n"
            "    tool: write_csv\n"
            '    config: {path: "{scratch}/kept.csv"}\n'
            "    inputs: {rows: kept.true}\n",
            encoding="utf8",
        )

        assert read_pipeline_file(written, rules=RULES) == load_pipeline(document, rules=RULES)

        source_root = Path(dantron_etl.__file__).parent.parent
        probe = subprocess.run(
            [sys.executable, "-c", "import dantron_etl.pipeline, sys; print('yaml' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(source_root),
            env={**_BARE_ENVIRONMENT, "PYTHONPATH": str(source_root), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert probe.stdout.strip() == "False", (
            "importing the pipeline module must not pull in a YAML parser: the dependency belongs to "
            f"the file-reading convenience alone (stderr: {probe.stderr})"
        )


# ---------------------------------------------------------------------------
# Escaping is the loader's job, not the author's
# ---------------------------------------------------------------------------

#: Values that are ordinary in a real column and awkward in a hand-edited file. Every one of them
#: has to survive the trip from a rules mapping into a compiled predicate without the person editing
#: the pipeline doing anything special -- which is the whole feature. The last three are the ones
#: that catch an implementation that is merely close: a value that already holds a doubled quote
#: must not be un-doubled, a value that reads like an expression must stay a value, and a value that
#: is nothing at all must still be a value.
AWKWARD_VALUES: Final[tuple[str, ...]] = (
    "O'Brian",
    'Parkside "North"',
    "back\\slash",
    "both ' and \" together",
    "café ✓",
    "  padded  ",
    "a''b",
    "' or true or '",
    "",
)


class TestARuleValueIsEscapedForTheAuthorRatherThanByThem:
    """A literal reaches the expression language correctly without anyone remembering to escape it.

    The person editing a pipeline file is not writing code. They are stating that the codes worth
    keeping are ``O'Brian`` and ``Parkside``, and they have no reason to know that one of those two
    values has to be spelled differently from the other. So the escaping is the loader's, and the
    property below is the one that matters: whatever the value, ``${rules.x}`` compiles to a
    predicate that matches exactly that value and nothing else.

    Two spellings are in use in real pipeline files and both must work. A reference standing on its
    own inside an expression -- ``[code] = ${rules.x}`` -- and a reference the author has already
    wrapped in quotes -- ``[code] = '${rules.x}'`` -- which is how every text rule in this
    repository's own pipelines is written today, and how the fixture at the top of this file is
    written. A value that is not text is wrapped in the *language's* own tag instead:
    ``date'${rules.start_month}'``. None of those three may break.
    """

    def _resolved(self, predicate: str, rules: Mapping[str, Any]) -> str:
        """The predicate of the fixture's filter step, once the loader has finished with it."""
        document = _document()
        _step(document, "kept")["config"]["predicate"] = predicate
        return load_pipeline(document, rules={**RULES, **rules}).step("kept").config["predicate"]

    def test_a_scalar_with_an_apostrophe_reaches_the_language_as_that_value(self) -> None:
        """Both spellings compile, and both match the row the author named.

        Unquoted, the loader has to supply the quotes; already quoted, it has to supply only the
        escape. Today the first splices a bare ``O'Brian`` into the text, where ``O`` is read as a
        name nobody defined, and the second closes the author's literal early.
        """
        rows = pl.DataFrame({"code": ["O'Brian", "OBrian", "O Brian", "Parkside"]})

        for predicate in ("[code] = ${rules.kept_code}", "[code] = '${rules.kept_code}'"):
            resolved = self._resolved(predicate, {"kept_code": "O'Brian"})
            assert rows.filter(compile_expr(resolved))["code"].to_list() == ["O'Brian"], resolved

    def test_a_list_is_written_in_this_languages_own_quoting_and_never_in_pythons(self) -> None:
        """``('a', 'b')`` as before, with each element escaped here rather than by ``repr``.

        Python's ``repr`` switches to double quotes the moment a value holds an apostrophe, and a
        double quote is refused by this language by design -- so a list of names fails at load the
        moment one of the names is Irish. Square brackets are not an option either: in an expression
        they mean a column.
        """
        resolved = self._resolved(
            "[code] in ${rules.kept_codes}",
            {"kept_codes": ["O'Brian", "Parkside"]},
        )

        assert '"' not in resolved, f"a double quote is not part of this language: {resolved!r}"
        rows = pl.DataFrame({"code": ["O'Brian", "OBrian", "Parkside", "Larkin"]})
        assert rows.filter(compile_expr(resolved))["code"].to_list() == ["O'Brian", "Parkside"]

    @pytest.mark.parametrize(
        ("predicate", "rules", "expected"),
        [
            ("[month] >= date'${rules.start_month}'", {"start_month": date(2020, 1, 1)}, "[month] >= date'2020-01-01'"),
            ("[span] > ${rules.max_span}", {"max_span": 60}, "[span] > 60"),
            ("abs([net]) <= ${rules.tolerance}", {"tolerance": 0.005}, "abs([net]) <= 0.005"),
            ("[qty] in ${rules.sizes}", {"sizes": [1, 2]}, "[qty] in (1, 2)"),
        ],
    )
    def test_a_rule_value_that_is_not_text_still_reads_as_it_reads_today(
        self, predicate: str, rules: Mapping[str, Any], expected: str
    ) -> None:
        """The shapes a real pipeline file actually uses, pinned exactly as they resolve now.

        A date is tagged by the author as ``date'...'`` and the value belongs *inside* that tag, not
        wrapped in quotes of its own. A count and a tolerance are spliced as numbers, because a
        number in quotes is text, and text compared against a number column is refused. A list of
        numbers stays a list of numbers for the same reason. Quoting everything would break all
        four, and each of them is load-bearing in a pipeline that runs today.
        """
        assert self._resolved(predicate, rules) == expected
        compile_expr(expected)

    def test_a_reference_that_is_the_whole_field_still_takes_the_rules_own_value_and_type(self) -> None:
        """Nothing is quoted, tagged or rendered when the reference *is* the field: the value is used.

        A list stays a list, a number stays a number, and a string arrives verbatim -- including a
        string that is itself an expression, which is how a whole predicate gets stated as a rule.
        That case also pins the direction of travel: the loader adds no quoting here, so the quoting
        in the value is the author's own and is read by the language exactly as written.
        """
        document = _document()
        _step(document, "entries")["config"]["skip_rows"] = "${rules.skipped_lines}"
        _step(document, "kept")["config"]["predicate"] = "${rules.kept_predicate}"

        spec = load_pipeline(
            document,
            rules={**RULES, "skipped_lines": 2, "kept_predicate": "[code] = 'O''Brian'"},
        )

        assert spec.step("entries").config["skip_rows"] == 2
        assert spec.step("paired").config["keys"] == ["entry_id"]
        assert spec.step("kept").config["predicate"] == "[code] = 'O''Brian'"
        rows = pl.DataFrame({"code": ["O'Brian", "OBrian"]})
        assert rows.filter(compile_expr(spec.step("kept").config["predicate"]))["code"].to_list() == ["O'Brian"]

    @pytest.mark.parametrize("value", AWKWARD_VALUES)
    def test_any_value_compiles_to_a_predicate_that_matches_exactly_that_value(self, value: str) -> None:
        """The joint property, stated once over the values most likely to break it.

        The frame holds every awkward value at once, so "matches exactly that value" is asserted
        against the others rather than against an empty room: ``a'b`` sits beside ``a''b`` so that
        an escape applied one time too many is caught, and ``' or true or '`` is in the list because
        a value that reads like a fragment of an expression must arrive as a value. A predicate that
        matched nothing would be indistinguishable, in a log, from a day on which nothing happened.
        """
        rows = pl.DataFrame({"code": [*AWKWARD_VALUES, "a'b", "OBrian", "O Brian"]})

        for predicate in ("[code] = ${rules.kept_code}", "[code] = '${rules.kept_code}'"):
            resolved = self._resolved(predicate, {"kept_code": value})
            assert rows.filter(compile_expr(resolved))["code"].to_list() == [value], resolved

    def test_a_value_nobody_could_have_escaped_is_matched_by_a_run_of_the_file(self, tmp_path: Path) -> None:
        """End to end: a file on disk, a rule with an apostrophe in it, and the row that came out.

        Everything above stops at a compiled expression. This one runs the flow, because the failure
        this feature exists to prevent is not an exception -- it is a filter that quietly keeps the
        wrong rows, or none, while the run reports success.
        """
        entries = tmp_path / "entries.csv"
        entries.write_text("code,amount\nO'Brian,10\nOBrian,20\nParkside,30\n", encoding="utf8")
        kept = tmp_path / "kept.csv"
        flow = tmp_path / "flow.yaml"
        flow.write_text(
            "name: a neutral fixture\n"
            "steps:\n"
            "  - id: entries\n"
            "    tool: read_csv\n"
            f"    config: {{path: '{entries}'}}\n"
            "  - id: kept\n"
            "    tool: filter\n"
            '    config: {predicate: "[code] = ${rules.kept_code}"}\n'
            "    inputs: {rows: entries.rows}\n"
            "  - id: published\n"
            "    tool: write_csv\n"
            f"    config: {{path: '{kept}'}}\n"
            "    inputs: {rows: kept.true}\n",
            encoding="utf8",
        )

        spec = read_pipeline_file(flow, rules={"kept_code": "O'Brian"})
        run_pipeline(spec)

        assert pl.read_csv(kept)["code"].to_list() == ["O'Brian"]


# ---------------------------------------------------------------------------
# Which spelling a rule value gets is the field's question, not the loader's
# ---------------------------------------------------------------------------


def _load_step(step: dict[str, Any], rules: Mapping[str, Any]) -> Mapping[str, Any]:
    """``step`` appended to the fixture flow, loaded, and its configuration as the loader left it.

    A sink is appended for any tool whose effect *is* one of its ports, so the step under test is
    never also a dead end and the refusal under test is the one the test is named for.
    """
    document = _document()
    document["steps"].append(step)
    if REGISTRY.get(step["tool"]).category not in {"output", "assert"}:
        document["steps"].append(_writer_step(f"{step['id']}_sink", f"{step['id']}.rows"))
    return load_pipeline(document, rules={**RULES, **rules}).step(step["id"]).config


class TestARuleValueIsSpelledForTheFieldItLandsIn:
    """Quoting belongs to the expression language, so only an expression field may receive it.

    A ``${rules.*}`` reference embedded in surrounding text is written in the expression language's
    own quoting -- ``O'Brian`` arrives as ``'O''Brian'`` -- and that is right, because the author is
    stating a value and has no reason to know how this language spells one. It is right *for an
    expression field*. A pattern, a path, a sheet name, a column name and a table name are none of
    them expressions, and a quote spliced into any of those is not an escape: it is a character in
    the value.

    The pattern case is the one that decides this. ``(?P<period>'\\d{4}-\\d{2}')`` is a valid regular
    expression. It compiles, it matches nothing, and the run reports success -- the exact failure
    MANUAL section 5 calls unacceptable. The path case is quieter still: the file is written, under
    a name nobody asked for, beside the one somebody is waiting for.

    The tool already knows which of its fields are expressions: ``config_template()`` declares
    ``type="expression"`` for those and ``type="string"`` for the rest, and that declaration is the
    whole seam. So each field below is asserted to receive the rule's value and nothing else --
    whatever the value contains, and at whatever depth the field sits.
    """

    def test_a_pattern_receives_the_value_and_stays_a_usable_pattern(self) -> None:
        """A regex field is not an expression, and a quote inside one is a character to match.

        Asserted by matching, not only by equality: a pattern that has been quoted still compiles,
        so an equality assertion alone would leave a reader unsure whether the difference mattered.
        It matches nothing, which is indistinguishable in a log from a day with no data.
        """
        config = _load_step(
            {
                "id": "periods",
                "tool": "regex_extract_groups",
                "config": {"source": "[label]", "pattern": "(?P<period>${rules.period_pattern})"},
                "inputs": {"rows": "kept.true"},
            },
            {"period_pattern": r"\d{4}-\d{2}"},
        )

        assert config["pattern"] == r"(?P<period>\d{4}-\d{2})"
        assert "'" not in config["pattern"], "a quote in a pattern is a character to match, not an escape"
        assert re.compile(config["pattern"]).search("period 2026-09 closed") is not None

    def test_a_file_path_receives_the_value_and_names_the_file_somebody_is_waiting_for(self) -> None:
        """An apostrophe in the value is part of the name, and quotes around it are not.

        Both halves are pinned here: no quotes are added, and the one quote the value carries is not
        doubled. A path is the field where getting this wrong is silent -- the write succeeds, and
        the file nobody asked for sits beside the one somebody is waiting for.
        """
        document = _document()
        _step(document, "published")["config"]["path"] = "{scratch}/${rules.holder}_figures.csv"

        config = load_pipeline(document, rules={**RULES, "holder": "O'Brian"}).step("published").config

        assert config["path"] == "{scratch}/O'Brian_figures.csv"

    def test_a_sheet_name_receives_the_value(self) -> None:
        """A workbook sheet is a name a reader sees, and quotes around it are not part of the name."""
        config = _load_step(
            {
                "id": "workbook",
                "tool": "write_excel",
                "config": {
                    "path": "{scratch}/figures.xlsx",
                    "sheets": [{"port": "rows", "sheet": "Figures ${rules.period}"}],
                },
                "inputs": {"rows": "paired.joined"},
            },
            {"period": "2026-09"},
        )

        assert config["sheets"][0]["sheet"] == "Figures 2026-09"

    def test_a_column_name_nested_in_an_array_of_objects_receives_the_value(self) -> None:
        """The seam has to reach nesting: the name sits at ``columns[0].name``, beside an expression.

        The same step carries both kinds of field, one element apart, so a fix that reads the top
        level of the template and stops is caught here rather than in a tool nobody uses. The
        expression beside it keeps the language's quoting, because that field really is one.
        """
        config = _load_step(
            {
                "id": "totals",
                "tool": "formula",
                "config": {
                    "columns": [
                        {
                            "name": "${rules.prefix}_total",
                            "value": "concat([code], '${rules.prefix}')",
                        }
                    ]
                },
                "inputs": {"rows": "kept.true"},
            },
            {"prefix": "O'Brian"},
        )

        assert config["columns"][0]["name"] == "O'Brian_total"
        assert config["columns"][0]["value"] == "concat([code], 'O''Brian')"

    def test_a_schema_and_a_table_receive_the_value_and_stay_plain_identifiers(self) -> None:
        """A database name is spliced into a statement, and a quote is not a name it may carry.

        This is the field where the current spelling does not even reach the run: the tool's own
        identifier check refuses ``'area'_marts`` at load. A refusal is the kindest of the four
        failures on this page, and it still means a configuration a person wrote correctly cannot
        be loaded at all.
        """
        config = _load_step(
            {
                "id": "ledger",
                "tool": "read_table",
                "config": {
                    "connection": "A_STORE_URI",
                    "schema": "${rules.area}_marts",
                    "table": "${rules.area}_entries",
                },
            },
            {"area": "northern"},
        )

        assert config["schema"] == "northern_marts"
        assert config["table"] == "northern_entries"


class TestAnExpressionFieldKeepsTheQuotingItAlreadyHas:
    """The other side of the same decision, including the expression fields that sit inside arrays.

    ``TestARuleValueIsEscapedForTheAuthorRatherThanByThem`` above pins the top-level expression
    field. These are the nested ones: a ladder's conditions and results, an aggregation's value.
    They are declared ``type="expression"`` exactly like ``filter.predicate`` is, they are where
    most expressions in a real flow actually live, and a fix that consults the template has to walk
    down to them or it turns a working file into one that matches nothing.
    """

    def test_a_ladders_conditions_and_results_keep_the_languages_quoting(self) -> None:
        """``cases[].when`` and ``cases[].then`` are expressions two levels down, and stay so."""
        config = _load_step(
            {
                "id": "banding",
                "tool": "case_when",
                "config": {
                    "cases": [
                        {
                            "when": "[code] = '${rules.kept_code}'",
                            "then": "concat('band ', '${rules.band}')",
                        }
                    ],
                    "otherwise": "'none'",
                    "output": "band",
                },
                "inputs": {"rows": "kept.true"},
            },
            {"kept_code": "O'Brian", "band": "O'Neill"},
        )

        assert config["cases"][0]["when"] == "[code] = 'O''Brian'"
        assert config["cases"][0]["then"] == "concat('band ', 'O''Neill')"
        rows = pl.DataFrame({"code": ["O'Brian", "OBrian"]})
        assert rows.filter(compile_expr(config["cases"][0]["when"]))["code"].to_list() == ["O'Brian"]

    def test_an_aggregations_value_keeps_the_languages_quoting(self) -> None:
        """``aggregations[].value`` is an expression beside two plain strings in the same object."""
        config = _load_step(
            {
                "id": "grouped",
                "tool": "group_by",
                "config": {
                    "by": ["code"],
                    "aggregations": [
                        {"name": "scaled", "function": "sum", "value": "[amount] * ${rules.factor}"},
                        {"name": "tagged", "function": "max", "value": "concat([code], '${rules.suffix}')"},
                    ],
                },
                "inputs": {"rows": "kept.true"},
            },
            {"factor": 2, "suffix": "O'Brian"},
        )

        assert config["aggregations"][0]["value"] == "[amount] * 2"
        assert config["aggregations"][1]["value"] == "concat([code], 'O''Brian')"

    def test_an_assertions_predicate_keeps_the_languages_quoting(self) -> None:
        """The tool whose whole job is to refuse a run must not be the one quietly asserting nothing."""
        config = _load_step(
            {
                "id": "checked",
                "tool": "assert_rows",
                "config": {"predicate": "[code] = '${rules.kept_code}'", "label": "codes are admitted"},
                "inputs": {"rows": "kept.true"},
            },
            {"kept_code": "O'Brian"},
        )

        assert config["predicate"] == "[code] = 'O''Brian'"
        rows = pl.DataFrame({"code": ["O'Brian", "OBrian"]})
        assert rows.filter(compile_expr(config["predicate"]))["code"].to_list() == ["O'Brian"]


class TestEachPlaceholderIsSpelledForWhereItSits:
    """Two references in one field can want opposite spellings, and each gets its own answer.

    Whether a reference needs the quotes supplied is decided by whether the author already wrote
    them around it, and a field may hold one of each. No test in this file has ever put two
    references in one string, so a loader that decided once for the whole field -- from the first
    reference it found -- would pass everything else here and turn one of the two into a name the
    language does not know, or into a quoted string compared against a number.
    """

    def _resolved(self, predicate: str, rules: Mapping[str, Any]) -> str:
        document = _document()
        _step(document, "kept")["config"]["predicate"] = predicate
        return load_pipeline(document, rules={**RULES, **rules}).step("kept").config["predicate"]

    def test_two_references_in_one_predicate_get_opposite_spellings(self) -> None:
        """One inside the author's quotes and one outside, in a single field, both compiling."""
        resolved = self._resolved(
            "[code] = '${rules.kept_code}' and [zone] = ${rules.zone}",
            {"kept_code": "O'Brian", "zone": "North"},
        )

        assert resolved == "[code] = 'O''Brian' and [zone] = 'North'"
        rows = pl.DataFrame(
            {
                "code": ["O'Brian", "O'Brian", "OBrian"],
                "zone": ["North", "South", "North"],
                "row": [1, 2, 3],
            }
        )
        assert rows.filter(compile_expr(resolved))["row"].to_list() == [1]

    def test_a_value_in_the_middle_does_not_shift_the_spelling_of_a_later_reference(self) -> None:
        """Three references, in/out/in, with the middle value carrying a quote of its own.

        The middle reference is spelled with quotes the author did not write, so a loader that
        decided the third reference's spelling by counting quotes in the text it had produced so
        far would be counting its own output. The parity that answers the question belongs to the
        field as the author wrote it.
        """
        resolved = self._resolved(
            "[code] = '${rules.kept_code}' and [zone] = ${rules.zone} and [tier] = '${rules.tier}'",
            {"kept_code": "O'Brian", "zone": "O'Neill", "tier": "gold"},
        )

        assert resolved == "[code] = 'O''Brian' and [zone] = 'O''Neill' and [tier] = 'gold'"
        rows = pl.DataFrame(
            {
                "code": ["O'Brian", "O'Brian", "O'Brian"],
                "zone": ["O'Neill", "O'Neill", "ONeill"],
                "tier": ["gold", "silver", "gold"],
                "row": [1, 2, 3],
            }
        )
        assert rows.filter(compile_expr(resolved))["row"].to_list() == [1]


class TestATruthValueOrANullIsSpelledInThisLanguage:
    """YAML is the likeliest source of a boolean rule, and Python's spelling of one is not read here.

    ``no``, ``off``, ``~`` and their case variants stop being words the moment a rules file is
    parsed, which MANUAL section 5 already warns about for keys and enum values. A rule *value* is
    the same trap: ``wanted: NO`` is the boolean false, and written into a predicate as ``False``
    it names nothing this language knows -- or, inside the author's quotes, becomes the four
    characters ``False`` and matches no row at all.

    So the two positions get different answers. Outside a literal a truth value and a null have a
    spelling in this language, and they get it. Inside a literal they have none: the author asked
    for text and the rules file does not hold text, which is a refusal naming the fix rather than a
    run that reports success over nothing.
    """

    def _resolved(self, predicate: str, rules: Mapping[str, Any]) -> str:
        document = _document()
        _step(document, "kept")["config"]["predicate"] = predicate
        return load_pipeline(document, rules={**RULES, **rules}).step("kept").config["predicate"]

    def test_a_truth_value_outside_a_literal_is_spelled_the_way_this_language_spells_one(self) -> None:
        resolved = self._resolved("[flag] = ${rules.enabled}", {"enabled": True})

        assert resolved == "[flag] = true"
        rows = pl.DataFrame({"flag": [True, False], "row": [1, 2]})
        assert rows.filter(compile_expr(resolved))["row"].to_list() == [1]

    def test_a_null_outside_a_literal_is_spelled_the_way_this_language_spells_one(self) -> None:
        resolved = self._resolved("coalesce([code], ${rules.fallback}) is null", {"fallback": None})

        assert resolved == "coalesce([code], null) is null"
        rows = pl.DataFrame({"code": ["A", None], "row": [1, 2]})
        assert rows.filter(compile_expr(resolved))["row"].to_list() == [2]

    def test_a_truth_value_where_the_author_asked_for_text_is_refused_by_name(self) -> None:
        """``wanted: NO`` against a text column resolves to ``'False'`` today, and matches nothing."""
        with pytest.raises(PipelineError) as raised:
            self._resolved("[status] = '${rules.wanted}'", {"wanted": False})

        message = str(raised.value)
        assert "wanted" in message, "the refusal must name the rule whose value is the problem"
        assert "kept" in message, "and the step, because a rule is read from many"
        assert "quote" in message.lower(), "the fix is a pair of quotes in the rules file"


# ---------------------------------------------------------------------------
# A fingerprint over the closure that produced a port
#
# The other half of cache-and-run: a run may be handed a frame a previous run left on a computed
# port, so the steps that produced it do not run again. That is only safe if the question "is the
# configuration that produced this port still the configuration in the file?" has a cheap, exact
# answer -- which is what a fingerprint is. It covers the ancestor closure of a step and the step
# itself, because a cached frame is the product of everything upstream of it, not of one config
# block.
#
# What it deliberately does not cover is lane, stage and why. None of the three can change a figure,
# and a cache that went stale because somebody improved a 'why' note is a cache nobody will trust --
# which is to say a cache everybody works around, and the mechanism is then worse than none.
# ---------------------------------------------------------------------------

#: A mutation of a fresh fixture document, applied before it is loaded. Every case below is written
#: as one of these so that the two documents being compared differ in exactly the stated way.
Mutation = Callable[[dict[str, Any]], None]


def _fingerprint_of(step_id: str, mutate: Mutation | None = None, rules: Mapping[str, Any] = RULES) -> str:
    """``step_id``'s fingerprint in the fixture flow, optionally mutated first."""
    document = _document()
    if mutate is not None:
        mutate(document)
    return load_pipeline(document, rules=rules).fingerprint(step_id)


def _rewire(document: dict[str, Any], step_id: str, port: str, source: str) -> None:
    """Point one input port of ``step_id`` at a different producer, changing no configuration."""
    _step(document, step_id)["inputs"][port] = source


def _rename(document: dict[str, Any], step_id: str, new_id: str) -> None:
    """Rename a step and every edge that names it, so the document stays loadable."""
    for step in document["steps"]:
        if step["id"] == step_id:
            step["id"] = new_id
        for port, source in dict(step.get("inputs", {})).items():
            producer, produced_port = source.split(".", 1)
            if producer == step_id:
                step["inputs"][port] = f"{new_id}.{produced_port}"


class _NoTextForm:
    """A configuration value with no canonical way of being written down.

    Its ``repr`` carries this process's memory address, which is the whole reason the fingerprint
    must not fall back on ``str``: the address differs between the run that wrote a fingerprint
    beside a frame and the run that compares it, so every cache would report itself stale and the
    failure would look like the feature not working rather than like a value nobody can write.
    """


def _spec_carrying(value: Any) -> PipelineSpec:
    """A one-step spec whose configuration holds ``value``, assembled rather than loaded.

    Assembled on purpose. Every field the shipped catalog declares is typed and none of the declared
    types is a date, so ``load_pipeline`` refuses a bare ``2026-04-01`` before a fingerprint could
    ever be taken of one -- while ``yaml.safe_load`` goes on handing back a ``datetime.date`` for
    exactly that line, and :class:`PipelineSpec` is a public dataclass a caller assembling a flow in
    memory may build directly. So a fingerprint has to answer for a value the JSON encoder has no
    spelling for, whether or not today's loader is the way it arrives, and what it answers is what
    the cases below pin.
    """
    return PipelineSpec(
        name="a neutral fixture",
        steps=(StepSpec(id="entries", tool="read_csv", config={"path": value}, inputs={}),),
    )


class TestAFingerprintAnswersWhetherAPortIsStillTheSamePort:
    """One string per step, over that step's ancestor closure, stable enough to be written down."""

    def test_a_fingerprint_is_a_stable_string_per_step(self) -> None:
        """Same document, same string; different steps, different strings.

        Stable is the whole requirement: the string is recorded beside a frame in one run and
        compared in another, so a fingerprint that changed between two loads of one document would
        make every cache stale the moment it was written, and one that did not change between two
        different steps would let a frame stand in for a port that never produced it.
        """
        first = load_pipeline(_document(), rules=RULES)
        second = load_pipeline(_document(), rules=RULES)

        fingerprint = first.fingerprint("paired")
        assert isinstance(fingerprint, str)
        assert fingerprint != ""
        assert fingerprint == first.fingerprint("paired"), "asking twice is asking the same question"
        assert fingerprint == second.fingerprint("paired"), "two loads of one document are one pipeline"

        distinct = {step.id: first.fingerprint(step.id) for step in first.steps}
        assert len(set(distinct.values())) == len(distinct), "no two ports of this flow share a fingerprint"

    @pytest.mark.parametrize(
        ("first", "second", "step_id"),
        [
            pytest.param(
                None,
                lambda document: _step(document, "paired").__setitem__("lane", "intake"),
                "paired",
                id="the swimlane the step is drawn in",
            ),
            pytest.param(
                None,
                lambda document: _step(document, "paired").__setitem__("stage", "collect"),
                "paired",
                id="the stage the step is filed under",
            ),
            pytest.param(
                None,
                lambda document: _step(document, "paired").__setitem__("why", "a better sentence"),
                "paired",
                id="a note on the step itself, improved",
            ),
            pytest.param(
                None,
                lambda document: _step(document, "kept").__setitem__("why", "a better sentence upstream"),
                "paired",
                id="a note on an ancestor, improved",
            ),
            pytest.param(
                lambda document: _step(document, "kept").__setitem__(
                    "config", {"predicate": "[code] = 'A'", "nulls": "include"}
                ),
                lambda document: _step(document, "kept").__setitem__(
                    "config", {"nulls": "include", "predicate": "[code] = 'A'"}
                ),
                "paired",
                id="the same configuration, its keys written in the other order",
            ),
        ],
    )
    def test_presentation_and_documentation_do_not_change_a_fingerprint(
        self,
        first: Mutation | None,
        second: Mutation | None,
        step_id: str,
    ) -> None:
        """A ruling, not an implementation detail: a fingerprint covers what can change a figure.

        Lane, stage and why are what a canvas draws and what a reader reads. Not one of them reaches
        a tool, so a frame produced under one 'why' is the identical frame under another.
        Invalidating a cache on them would punish exactly the edit this codebase most wants people to
        make -- and an author who learns that documenting a step throws away an hour of upstream work
        stops documenting steps.

        The last case is the same ruling one level down. A mapping has no order, so a configuration
        whose keys were retyped in a different sequence is the same configuration; a fingerprint
        taken over the text of a dict would disagree, and would go stale on a reformat.
        """
        assert _fingerprint_of(step_id, first) == _fingerprint_of(step_id, second)

    @pytest.mark.parametrize(
        ("mutate", "step_id"),
        [
            pytest.param(
                lambda document: _step(document, "paired").__setitem__("config", {"keys": ["entry_id"], "how": "left"}),
                "paired",
                id="the step's own configuration",
            ),
            pytest.param(
                lambda document: _step(document, "kept").__setitem__("config", {"predicate": "[code] = 'B'"}),
                "paired",
                id="a direct ancestor's configuration -- same step, same config, different ancestry",
            ),
            pytest.param(
                lambda document: _step(document, "entries").__setitem__("config", {"path": "{scratch}/other.csv"}),
                "paired",
                id="an ancestor two steps up, which is where a cached frame's rows came from",
            ),
            pytest.param(
                lambda document: _step(document, "paired").__setitem__("config", {"keys": ["entry_id", "header_code"]}),
                "paired",
                id="one more element in an array the configuration carries",
            ),
            pytest.param(
                lambda document: _rewire(document, "published", "rows", "paired.left_only"),
                "published",
                id="a rewired edge, with every configuration left alone",
            ),
            pytest.param(
                lambda document: _rename(document, "entries", "records"),
                "kept",
                id="an ancestor renamed, which renames the edge that reads it",
            ),
        ],
    )
    def test_a_change_anywhere_in_the_closure_changes_the_fingerprint(self, mutate: Mutation, step_id: str) -> None:
        """Everything upstream had a hand in the rows on this port, so everything upstream counts.

        The second case is the one the whole mechanism rests on. 'paired' is untouched -- same id,
        same tool, same configuration, same edges -- and a frame cached from it is nonetheless wrong,
        because the filter feeding it now keeps different rows. A fingerprint over the step alone
        would report a match and hand back yesterday's numbers under today's rules.

        The fifth case is the same argument for the wiring rather than the configuration: nothing is
        configured differently, the closure holds the same steps, and the port being read has moved
        from the rows that paired to the rows that did not.
        """
        assert _fingerprint_of(step_id, mutate) != _fingerprint_of(step_id)

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda document: _step(document, "published").__setitem__("config", {"path": "{scratch}/else.csv"}),
                id="a step downstream of the cached one is edited",
            ),
            pytest.param(
                lambda document: document["steps"].extend(
                    [_filter_step("audited", "entries.rows"), _writer_step("filed", "audited.true")]
                ),
                id="a whole new branch is added elsewhere in the flow",
            ),
        ],
    )
    def test_a_change_outside_the_closure_leaves_a_fingerprint_alone(self, mutate: Mutation) -> None:
        """This is the half that makes resuming worth anything at all.

        An author caches the first ten steps precisely so they can write the eleventh, run it, look
        at it, and write it again. A fingerprint that moved when anything in the document moved would
        go stale on the very edit the cache was taken to support, and the feature would save nobody
        anything. So the question it answers is narrow on purpose: what produced *this* port.
        """
        assert _fingerprint_of("kept", mutate) == _fingerprint_of("kept")

    def test_a_rule_value_the_configuration_quotes_is_part_of_the_fingerprint(self) -> None:
        """Rules are edited far oftener than wiring is, and they change figures.

        A rule is resolved into a configuration at load, so a pipeline run under a different rules
        file is a different pipeline however identical the document looks. A cached frame taken under
        the old value is exactly the wrong number this mechanism exists to refuse -- and nothing on
        the face of the document would show it, because the document still reads '${rules.kept_code}'.

        'headers' quotes no rule, and is the control: its fingerprint must not move for a rule it
        never reads, or every cache in a large flow would be thrown away by any edit to any rule.
        """
        other = {**RULES, "kept_code": "B"}

        assert _fingerprint_of("kept", rules=other) != _fingerprint_of("kept")
        assert _fingerprint_of("headers", rules=other) == _fingerprint_of("headers")

    def test_a_step_the_pipeline_does_not_have_is_refused_by_name(self) -> None:
        """Refused the way ``step`` already refuses one, because it is the same mistake.

        A fingerprint is asked for by a caller writing a cache index -- a step id in a dictionary
        somewhere, typed by hand. Returning some hash of nothing for a name that does not exist would
        produce an index whose entries can never match anything, and a run that silently recomputed
        everything while reporting that it had resumed.
        """
        spec = load_pipeline(_document(), rules=RULES)

        with pytest.raises(PipelineError) as raised:
            spec.fingerprint("pared")
        assert "pared" in str(raised.value)
        assert "paired" in str(raised.value), "the refusal lists the ids there actually are"

        with pytest.raises(PipelineError) as by_step:
            spec.step("pared")
        assert str(by_step.value) == str(raised.value), "one mistake, one wording, whichever way it was made"

    def test_a_fingerprint_is_the_same_string_in_another_process(self, tmp_path: Path) -> None:
        """The trap this test exists for is ``hash()``, which is salted per process for strings.

        A cache written by this morning's run and read by this afternoon's is the entire use case,
        and those are two processes. An implementation built on ``hash()`` passes every assertion
        above -- the string is stable within a run, differs between steps, ignores 'why' -- and then
        reports every cache stale the moment the interpreter restarts, which is the failure that
        looks like the feature not working rather than like a bug.

        Two child processes with deliberately different hash seeds, both compared against this one.
        The document travels as JSON so the children load the very bytes this test loaded, rather
        than a second copy of the fixture that could drift away from it.
        """
        document_path = tmp_path / "document.json"
        rules_path = tmp_path / "rules.json"
        document_path.write_text(json.dumps(_document()), encoding="utf8")
        rules_path.write_text(json.dumps(RULES), encoding="utf8")

        probe = tmp_path / "probe.py"
        probe.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "import json\n"
            "from dantron_etl.pipeline import load_pipeline\n"
            "document = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))\n"
            "rules = json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))\n"
            "print(load_pipeline(document, rules=rules).fingerprint(sys.argv[3]))\n",
            encoding="utf8",
        )

        source_root = Path(dantron_etl.__file__).parent.parent
        elsewhere: list[str] = []
        for seed in ("0", "1"):
            result = subprocess.run(
                [sys.executable, str(probe), str(document_path), str(rules_path), "paired"],
                capture_output=True,
                text=True,
                check=True,
                cwd=str(tmp_path),
                env={
                    **_BARE_ENVIRONMENT,
                    "PYTHONPATH": str(source_root),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": seed,
                },
            )
            elsewhere.append(result.stdout.strip())

        here = load_pipeline(_document(), rules=RULES).fingerprint("paired")
        assert elsewhere[0] == here, "a fingerprint must survive a process boundary"
        assert elsewhere[1] == here, "and must not depend on the interpreter's per-process hash seed"

    @pytest.mark.parametrize(
        ("first", "second", "same"),
        [
            pytest.param(date(2026, 4, 1), date(2026, 4, 1), True, id="one cutover date, stated twice"),
            pytest.param(date(2026, 4, 1), date(2026, 4, 2), False, id="two cutover dates one day apart"),
            pytest.param(date(2026, 4, 1), date(2026, 5, 1), False, id="the month a period cutover moves on to"),
            pytest.param(date(2026, 4, 1), "2026-04-01", True, id="a date, and the same date quoted as text"),
            pytest.param(
                date(2026, 4, 1),
                datetime(2026, 4, 1),
                False,
                id="a day, and the timestamp at the start of that day",
            ),
        ],
    )
    def test_a_date_in_a_configuration_is_fingerprinted_by_the_moment_it_names(
        self,
        first: Any,
        second: Any,
        same: bool,
    ) -> None:
        """A period cutover is exactly the configuration a monthly build edits between two runs.

        ``as_of: 2026-04-01``, unquoted, is a ``datetime.date`` the moment YAML has read it -- not a
        string -- and it is the one value a finance pipeline is certain to change on the first of
        the month. If two dates a month apart fingerprinted alike, a cache taken in April would be
        accepted in May and the resumed run would publish April's figures under May's configuration:
        the run succeeds, nothing is refused, and the number is wrong in the one way this whole
        mechanism exists to catch. So the day the value names has to reach the hash, which means
        writing it down canonically rather than reaching for its type's name or for a constant.

        The fourth case states a ruling rather than a mechanism. A date's canonical form *is* its
        ISO text, so the unquoted value and the quoted one are the same day said two ways and are
        deliberately one fingerprint -- adding a pair of quotes to a pipeline file changes no figure
        and must not throw away an author's cache. The fifth is the other half of that: a timestamp
        is a date too, and a value carrying a time of day is not the day it falls on.
        """
        taken = _spec_carrying(first).fingerprint("entries")
        again = _spec_carrying(second).fingerprint("entries")

        assert (taken == again) is same

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            pytest.param({"north", "south"}, "set", id="a set, which no configuration file can round-trip"),
            pytest.param(_NoTextForm(), "_NoTextForm", id="an object whose repr carries a memory address"),
        ],
    )
    def test_a_value_with_no_canonical_text_form_is_refused_rather_than_stringified(
        self,
        value: Any,
        named: str,
    ) -> None:
        """Falling back on ``str`` here would be the failure that looks like the feature not working.

        ``str`` of an arbitrary object is a repr carrying a memory address, so the fingerprint taken
        beside a frame in one process and the fingerprint compared in the next would differ for a
        configuration nobody touched. Every cache would then report itself stale, an author would
        conclude that resuming a run simply does not work, and nothing anywhere would say why. A set
        is the same problem one step less obviously: it has a text form, and the form it has depends
        on an iteration order no configuration file can promise.

        So the value is refused instead, and the refusal is the useful kind -- it names the type it
        could not write down and says what to do about it, which is to quote the value in the
        pipeline or rules file. That sentence is the whole difference between an author fixing one
        line and an author abandoning the cache.
        """
        with pytest.raises(PipelineError) as raised:
            _spec_carrying(value).fingerprint("entries")

        message = str(raised.value)
        assert named in message, "the refusal names the type it could not write down"
        assert "fingerprint" in message, "and says which question it was that could not be answered"
        assert "text" in message, "and names the fix: write the value as text in the file"


# ---------------------------------------------------------------------------
# Containers: a title over a handful of steps
#
# Six hundred and seventy-two boxes on a canvas cannot be followed. A few dozen titled containers
# can -- "fetch the crm accounts and put the header row back" says in one line what six boxes
# say in six, and a reader who trusts that line never has to open it.
#
# The shape is a `group` field on the step and a `groups:` block giving each id a title. Steps stay
# in one flat list, so a step moves between containers by editing one line and nothing about the
# file's shape changes. It is exactly how `lane` and `stage` already work, which is the reason it was
# chosen: a second nesting level in the document would make every tool that reads a pipeline file
# learn about it, for a fact that only a canvas cares about.
#
# What this section pins:
#
#   * a group is declared with a title, and a step names one by id;
#   * a step naming a group nobody declared is refused by name, and a declared group nobody joins is
#     refused too -- both of those are typos whose punishment would otherwise be a box that silently
#     is not drawn, or one drawn empty;
#   * a group's steps share a lane, because a lane is a horizontal band and a box across two bands
#     cannot be drawn honestly;
#   * a group's steps need *not* feed one another (argued at the case itself);
#   * a container's boundary -- what entered it and what left -- is a question about the wiring, so
#     the graph answers it and the page only renders the answer.
# ---------------------------------------------------------------------------

#: The title the cases below declare. Written as a sentence, because a title is the whole point: it
#: is what a reader gets instead of the six boxes, and an id is not a title.
SHAPE_TITLE: Final[str] = "Keep the codes the rules admit and pair them with their headers"


def _grouped_document() -> dict[str, Any]:
    """The fixture flow with its three shaping steps boxed into one container.

    ``kept`` reads from outside the container and ``published`` writes from inside it, so the
    container has a real boundary in both directions rather than being the whole flow.
    """
    document = _document()
    document["groups"] = [{"id": "shape", "title": SHAPE_TITLE}]
    for step_id in ("kept", "paired", "published"):
        _step(document, step_id)["group"] = "shape"
    return document


class TestStepsAreBoxedIntoContainers:
    """A group is a title over a handful of steps, and everything that can be refused before a run."""

    def test_a_document_declaring_groups_loads_them_with_their_titles_and_members(self) -> None:
        """The whole surface: the declared containers, the title of each, and who is in them.

        ``Group`` is imported inside the test rather than at the top of this file, so that a suite
        run before containers exist fails on the cases that are about them instead of collecting
        nothing at all.
        """
        from dantron_etl.pipeline import Group

        spec = load_pipeline(_grouped_document(), rules=RULES)

        assert spec.groups == (Group("shape", SHAPE_TITLE),)
        assert spec.group("shape").title == SHAPE_TITLE
        # Members in the order the steps were declared, which is the order the file reads in and the
        # only order a container has: a set would draw its title over a different arrangement on
        # every render, and a reader comparing two renders would be comparing two drawings.
        assert spec.group_members("shape") == ("kept", "paired", "published")
        assert spec.step("kept").group == "shape"
        # The property that lets containers be adopted one stage at a time rather than all at once:
        # a step that names no group is a step nothing changed about.
        assert spec.step("entries").group is None
        assert spec.group_members("shape") == tuple(s.id for s in spec.steps if s.group == "shape")

        # Named by id and refused by name, the way a missing step already is -- a reader who asks
        # for a container that is not there is told which ones there are.
        with pytest.raises(PipelineError) as raised:
            spec.group("shpae")
        assert "shpae" in str(raised.value) and "shape" in str(raised.value)

    def test_a_group_is_presentation_and_never_changes_what_a_step_computes(self) -> None:
        """The same ruling lane, stage and why already live under, and for the same reason.

        A container is a line drawn round some boxes so a person can read the canvas. It reaches no
        tool, so a frame produced inside a container is the identical frame outside one. Put `group`
        into the fingerprint and boxing three steps together throws away every cached frame behind
        them -- which is to say the first person to tidy the canvas pays for it with an hour of
        re-running, and the second person does not tidy the canvas.
        """
        plain = load_pipeline(_document(), rules=RULES)
        boxed = load_pipeline(_grouped_document(), rules=RULES)

        assert boxed.step("paired").group == "shape", "the group did not survive the load at all"
        for step_id in ("kept", "paired", "published"):
            assert plain.fingerprint(step_id) == boxed.fingerprint(step_id), (
                f"drawing a box round {step_id} invalidated its cache, so tidying the canvas costs a re-run"
            )

    def test_a_step_naming_a_group_nobody_declared_is_refused_by_name(self) -> None:
        """The typo this refusal exists for draws no container and says nothing about why.

        A misspelt group is not a step that fails: it is a step that quietly falls out of the box it
        was meant to be in, leaving a container one step short and a stray node beside it. On a flow
        with six hundred steps nobody finds that by looking. So it is refused the way an undeclared
        lane already is, naming the spelling that was written and the ones that exist.
        """
        document = _grouped_document()
        _step(document, "paired")["group"] = "shpae"

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "shpae" in message, "the refusal does not name the spelling that was written"
        assert "paired" in message, "the refusal does not name the step to go and fix"
        assert "shape" in message, "the refusal must list the groups there are"

    def test_a_group_declared_without_a_title_is_refused(self) -> None:
        """An id is not a title, and a container with nothing to say is worse than no container.

        The entire value of this feature is a reader seeing one sentence instead of six boxes. A
        group that falls back on its own id gives them `sf_acct_hdr_join` over six boxes, which is
        the same canvas with an extra line drawn on it -- and it is a silent failure, because the box
        is drawn, it looks finished, and nothing says a sentence was owed.

        Blank counts as missing. A title of spaces satisfies every test that asks whether the key is
        there, and draws a box with nothing written on it.
        """
        for missing in ({"id": "shape"}, {"id": "shape", "title": ""}, {"id": "shape", "title": "   "}):
            document = _grouped_document()
            document["groups"] = [missing]

            with pytest.raises(PipelineError) as raised:
                load_pipeline(document, rules=RULES)

            message = str(raised.value)
            assert "shape" in message, f"{missing}: the refusal does not name the group that owes a title"
            assert "title" in message.lower(), f"{missing}: the refusal does not say what is missing"

    def test_a_group_no_step_belongs_to_is_refused(self) -> None:
        """A declared group nobody joins is a typo or a leftover, and either way it draws an empty box.

        It is the mirror of the refusal above it, and the pair is the point: a group named by a step
        and not declared is refused, and a group declared and named by no step is refused. Pin only
        one side and the other spelling of the same typo -- fixing the `groups:` block instead of the
        step, or the step instead of the block -- passes the loader and shows a reader an empty
        rectangle with a title over nothing.
        """
        document = _grouped_document()
        document["groups"].append({"id": "orphan", "title": "Something that used to be here"})

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "orphan" in message, "the refusal does not name the group that has no steps"
        assert "shape" not in message, "the container that does have steps is not the problem"

    def test_a_group_whose_steps_sit_in_two_lanes_is_refused(self) -> None:
        """A lane is a horizontal band, and one box across two bands cannot be drawn honestly.

        Whatever the canvas did with it would be a lie: stretched over both bands it swallows every
        step of the intervening lane that is not in the container, and drawn round one band it leaves
        a member outside the box that claims to hold it. Neither of those says anything is wrong, and
        both are read as the drawing rather than as the file.

        Refused rather than silently redrawn, because the fix is one line -- move the step, or split
        the container in two -- and only the author knows which of the two they meant.
        """
        document = _grouped_document()
        _step(document, "entries")["group"] = "shape"  # lane 'intake', among three steps in 'shaping'

        with pytest.raises(PipelineError) as raised:
            load_pipeline(document, rules=RULES)

        message = str(raised.value)
        assert "shape" in message, "the refusal does not name the container that spans two lanes"
        assert "intake" in message and "shaping" in message, "the refusal names neither of the lanes it spans"

    def test_a_groups_steps_need_not_feed_one_another(self) -> None:
        """Any set of steps in one lane may be boxed together; connectedness is not required.

        The tempting rule is the opposite one -- a container should be a connected piece of the flow,
        since that is what "one job" looks like. It is the wrong rule, and the counter-example is the
        commonest container there is: *read the three workbooks*. Three reads that touch nothing of
        each other and meet for the first time in a join outside the box are a container a reader
        wants, and a connectedness rule refuses exactly it.

        What connectedness was really protecting is a different property, and it belongs to the
        canvas rather than to the loader: a box must enclose its own steps and nothing else. That is
        a question about where things are drawn -- a *connected* group can enclose a stranger just as
        easily -- so it is pinned where it can actually be checked, over the rendered markup.
        """
        document = _grouped_document()
        document["groups"] = [{"id": "sources", "title": "Read both extracts as they arrived"}]
        for step_id in ("kept", "paired", "published"):
            _step(document, step_id).pop("group")
        for step_id in ("entries", "headers"):  # two sources, one lane, no edge between them
            _step(document, step_id)["group"] = "sources"

        spec = load_pipeline(document, rules=RULES)

        assert spec.group_members("sources") == ("entries", "headers")

    def test_a_container_names_what_entered_it_and_what_left_it(self) -> None:
        """The boundary, which is a question about the wiring and so is answered by the graph.

        *In* is every port produced outside the container and read inside it, each named once however
        many members read it: two steps taking the same upstream port are one lot of rows arriving,
        and counting it twice would report a container that invented rows.

        *Out* is every port of a member that is read by a non-member **or by nobody at all**. The
        second half is the part that is easy to leave out and wrong to leave out: rows that stop at a
        member's port have still left the container -- they went to a file, to a report, or nowhere
        -- and a container whose last step is a write would otherwise claim nothing came out of it.

        What is excluded is the ports that are used up inside: `kept.true` goes to `paired`, which is
        in the same box, so it is the container's own plumbing and no part of what it published. An
        implementation that simply sums every member's ports gets a bigger number that looks
        perfectly plausible, which is why this is pinned port by port rather than as a total.
        """
        spec = load_pipeline(_grouped_document(), rules=RULES)

        assert spec.group_inputs("shape") == (PortRef("entries", "rows"), PortRef("headers", "rows"))
        assert spec.group_outputs("shape") == (
            PortRef("kept", "false"),  # read by nobody: the rows the container set aside
            PortRef("paired", "left_only"),
            PortRef("paired", "right_only"),
            PortRef("published", "rows"),
        )

        # Named once, not twice: `kept` and `also` both read the same upstream port, and that is one
        # lot of rows entering the container.
        shared = _grouped_document()
        shared["steps"].append({**_filter_step("also", "entries.rows"), "group": "shape"})
        shared["steps"].append({**_writer_step("also_out", "also.true"), "group": "shape"})

        widened = load_pipeline(shared, rules=RULES)

        assert widened.group_inputs("shape").count(PortRef("entries", "rows")) == 1
        assert PortRef("also", "true") not in widened.group_outputs("shape"), (
            "a port read by another member of the same container is the container's own plumbing"
        )
        assert PortRef("also", "false") in widened.group_outputs("shape")
