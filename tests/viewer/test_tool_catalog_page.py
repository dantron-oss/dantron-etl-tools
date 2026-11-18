"""Tests for the generated tool catalog: the page has to say everything the registry declares.

``scripts/tool_catalog_page.py`` turns ``REGISTRY`` into two artefacts -- a page a person browses
and a JSON twin a model reads -- and the page is the product: somebody who did not write the tools
assembles a flow by copying a worked example off it.

``check_tool`` already guarantees a great deal about the tools themselves. Every worked example is
executed against its tool, every configuration field is typed and described, every name and
category is checked. What none of that guarantees is that the catalog *says* all of it. A generator
that quietly dropped a field, a port, an example, or a whole tool would produce a page that looks
entirely correct and is incomplete, and every other test in the suite would still be green. That is
the same shape of defect as a drifted worked example, which this library has already decided is
worse than no example at all -- a reader trusts what is in front of them, and silence reads as
"there is nothing more to know" rather than "this was lost on the way out".

So these tests compare the catalog against the registry it was generated from, field by field and
frame by frame, and everything they expect is derived from ``REGISTRY`` rather than listed here. A
hardcoded list of the nineteen tools that ship today would rot on the day a twentieth arrived, and
catching what somebody forgot is the entire job.

Four properties, in the order they matter:

**Completeness.** Every registered tool appears and nothing else does; per tool, every port, every
configuration field, every worked example. A field silently dropped is the defect most likely to
happen and least likely to be noticed, because nothing about the surrounding page looks wrong.

**Depth.** ``read_csv``'s ``column_types`` and ``case_when``'s ``rules`` are arrays of objects, and
a generator that walked only the top level would describe such a field as "an array" and stop --
precisely the case a reader most needs spelled out, because it is the one they cannot guess. The
nested claim is made over every nested field the registry declares, not over those two by name.

**Fidelity of the examples.** A worked example's frames keep their column names, their dtypes and
their values, on both the input and the output side, over every port the example documents. The
dtypes are not decoration: a reader copying an example needs to know that a column came back as
text.

**One source for both artefacts.** The page and the JSON are not compared to each other -- one is
markup and one is data, and comparing them would mean parsing HTML, which pins layout. The honest
formulation is that ``main`` renders the page from the very value it writes as JSON, so the
human-facing artefact and the model-facing one cannot disagree about a tool without both being
wrong together.

What is deliberately *not* here. These tests make no claim about markup, CSS or layout: the page's
rendering is worked on independently, and a test that pinned a class name would be brittle for no
gain. The single structural claim -- that every tool's name and one-line description appear in the
page somewhere -- is the narrowest form of "a tool that exists in the data was actually rendered",
and it survives any restyling. Nor is ``check_registry`` called here: ``tests/core/test_registry.py`` already
checks every shipped tool against it, and repeating that would mean this file goes red for defects
that are not the catalog's.
"""

import html
import json
import importlib.util
from typing import Any

import pytest

from dantron_etl.paths import project_root
from dantron_etl.tool import REGISTRY, ConfigField, Tool, catalog_input_ports

PROJECT_ROOT = project_root(__file__)

# ``scripts`` is not an importable package -- it has no ``__init__`` and is not on the path -- so
# the module under test is loaded from its file, by a path resolved from the project root rather
# than relative to this file. One load for the whole module: executing it has no side effects
# beyond importing the registry, and ``main`` (the only function that writes anything) is called
# below with its output paths redirected.
_SPEC = importlib.util.spec_from_file_location(
    "library_tests.tool_catalog_page", PROJECT_ROOT / "scripts" / "tool_catalog_page.py"
)
assert _SPEC is not None and _SPEC.loader is not None, "tool_catalog_page.py is not where the project root says"
catalog_page = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(catalog_page)


@pytest.fixture(scope="module")
def data() -> dict[str, Any]:
    """The catalog as plain data, generated once from the live registry."""
    return catalog_page.catalog()


@pytest.fixture(scope="module")
def by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each tool's catalog entry, keyed by the tool's machine name."""
    return {entry["name"]: entry for entry in data["tools"]}


def _registered() -> tuple[Tool, ...]:
    """Every registered tool. A guard, so no test below can pass over an empty catalog."""
    tools = REGISTRY.tools()
    assert tools, "the registry is empty, so every completeness claim here would hold vacuously"
    return tools


def _check_field(spec: ConfigField, rendered: dict[str, Any], where: str) -> int:
    """Compare one configuration field against its catalog entry; return how many were compared.

    Recursive, because a field describing an array of objects only says something useful if the
    element's own members are described too.
    """
    assert rendered["type"] == spec.type, f"{where}: type not carried"
    assert rendered["description"] == spec.description, f"{where}: description not carried"
    assert bool(rendered["required"]) == spec.required, f"{where}: required not carried"
    assert bool(rendered.get("non_empty")) == bool(spec.non_empty), f"{where}: non-empty not carried"

    if spec.default is None:
        assert "default" not in rendered, f"{where}: a default the tool does not declare"
    else:
        assert rendered.get("default") == spec.default, f"{where}: default not carried"
    if spec.enum:
        assert rendered.get("allowed") == list(spec.enum), f"{where}: allowed values not carried"
    else:
        assert not rendered.get("allowed"), f"{where}: allowed values invented"

    compared = 1
    assert (spec.items is not None) == ("items" in rendered), f"{where}: element description lost or invented"
    if spec.items is not None:
        compared += _check_field(spec.items, rendered["items"], f"{where}[]")
    assert bool(spec.fields) == bool(rendered.get("fields")), f"{where}: member descriptions lost or invented"
    if spec.fields:
        assert list(rendered["fields"]) == list(spec.fields), f"{where}: member names lost"
        for member, inner in spec.fields.items():
            compared += _check_field(inner, rendered["fields"][member], f"{where}.{member}")
    return compared


def _declared_depth(spec: ConfigField) -> int:
    """Levels of declared shape: 1 for a scalar, 3 for an array of objects with typed members."""
    inner = [_declared_depth(spec.items)] if spec.items is not None else []
    inner += [_declared_depth(member) for member in (spec.fields or {}).values()]
    return 1 + max(inner, default=0)


def _rendered_depth(rendered: dict[str, Any]) -> int:
    """The same measure, taken on the catalog entry rather than on the tool's declaration."""
    inner = [_rendered_depth(rendered["items"])] if "items" in rendered else []
    inner += [_rendered_depth(member) for member in (rendered.get("fields") or {}).values()]
    return 1 + max(inner, default=0)



def _as_json(value: Any) -> str:
    """``value`` as JSON text. Dates are written as ISO text, the way the catalog's JSON file holds them."""
    return json.dumps(value, default=lambda v: v.isoformat())

class TestCatalogCompleteness:
    """The catalog says everything the registry declares, for every tool the registry holds."""

    def test_catalog_lists_every_registered_tool_and_nothing_else(self, data: dict[str, Any]) -> None:
        """A tool that is registered is on the page; a tool that is not registered is not."""
        listed = [entry["name"] for entry in data["tools"]]

        assert sorted(listed) == sorted(REGISTRY.names()), "the catalog and the registry disagree about what ships"
        assert len(listed) == len(set(listed)), f"a tool is catalogued twice: {listed}"
        assert data["tool_count"] == len(REGISTRY.names()), "the headline count disagrees with what is catalogued"

    def test_each_tool_keeps_its_ports_and_everything_it_says_about_itself(
        self, by_name: dict[str, dict[str, Any]]
    ) -> None:
        """Ports, category, and all four kinds of prose reach the catalog for every tool.

        The input ports are asked for the way the page asks for them, through
        ``catalog_input_ports``. For the twenty-six tools with a static attribute the two are the
        same question. For a tool that works its ports out from a step's configuration the attribute
        is empty by construction, and reading it here would have this test insist the page show
        nothing -- which is the one thing the page must not do.
        """
        for tool in _registered():
            entry = by_name[tool.name]
            helped = tool.help()

            assert entry["input_ports"] == list(catalog_input_ports(tool)), f"{tool.name}: input ports lost"
            assert entry["output_ports"] == list(tool.output_ports), f"{tool.name}: output ports lost"
            assert entry["category"] == tool.category, f"{tool.name}: category lost"
            assert entry["description"] == tool.description, f"{tool.name}: one-line description lost"
            assert entry["detail"] == helped.detailed_description.strip(), f"{tool.name}: detailed description lost"
            assert entry["notes"] == list(helped.notes), f"{tool.name}: notes lost"
            assert entry["field_help"] == dict(helped.field_help), f"{tool.name}: per-field help lost"

    def test_every_configuration_field_is_described_in_full(self, by_name: dict[str, dict[str, Any]]) -> None:
        """Every field of every template, with type, description, required, default, allowed, non-empty."""
        compared = 0
        for tool in _registered():
            template = tool.config_template()
            config = by_name[tool.name]["config"]

            assert list(config) == list(template), f"{tool.name}: configuration fields lost, added or reordered"
            for name, spec in template.items():
                compared += _check_field(spec, config[name], f"{tool.name}.{name}")

        assert compared > len(REGISTRY.names()), "too few fields compared for this to be checking anything"

    def test_nested_element_shape_survives_into_the_catalog(self, by_name: dict[str, dict[str, Any]]) -> None:
        """An array of objects is described down to its members, not stopped at "an array".

        ``_check_field`` already recurses, so the work here is refusing to let that recursion go
        vacuous: the corpus really does contain fields with nested shape -- ``read_csv``'s
        ``column_types`` and ``case_when``'s ``rules`` among them -- and the catalog reaches the
        bottom of each one.
        """
        nested: list[str] = []
        for tool in _registered():
            config = by_name[tool.name]["config"]
            for name, spec in tool.config_template().items():
                declared = _declared_depth(spec)
                if declared == 1:
                    continue
                nested.append(f"{tool.name}.{name}")
                assert _rendered_depth(config[name]) == declared, (
                    f"{tool.name}.{name}: the catalog flattens a field the tool describes {declared} levels deep"
                )

        assert len(nested) >= 2, f"expected several nested fields to check, found {nested}"

    def test_every_worked_example_keeps_its_frames_columns_dtypes_and_values(
        self, by_name: dict[str, dict[str, Any]]
    ) -> None:
        """Each example, each port it documents, each column name, dtype and value, in and out."""
        frames = 0
        for tool in _registered():
            examples = tool.help().examples
            rendered = by_name[tool.name]["examples"]

            assert len(rendered) == len(examples), f"{tool.name}: worked examples lost"
            for example, shown in zip(examples, rendered, strict=True):
                assert shown["scenario"] == example.scenario, f"{tool.name}: scenario lost"
                assert shown["configuration"] == dict(example.configuration), f"{tool.name}: example config lost"
                assert shown["explanation"] == example.explanation, f"{tool.name}: explanation lost"
                assert shown["row_order_significant"] == example.check_row_order, f"{tool.name}: order claim lost"

                for side, ports in (("inputs", example.inputs), ("outputs", example.outputs)):
                    assert list(shown[side]) == list(ports.names), f"{tool.name}: {side} ports lost"
                    for port, frame in ports.items():
                        where = f"{tool.name}.{side}.{port}"
                        shape = {name: str(dtype) for name, dtype in frame.schema.items()}
                        assert shown[side][port]["schema"] == shape, f"{where}: columns or dtypes lost"
                        assert shown[side][port]["rows"] == frame.to_dicts(), f"{where}: values lost"
                        frames += 1

        assert frames > len(REGISTRY.names()), "too few example frames compared for this to be checking anything"


class TestCatalogArtefacts:
    """The two artefacts are one value rendered twice, and rendering it is repeatable."""

    def test_generating_the_catalog_twice_produces_the_same_bytes(self, data: dict[str, Any]) -> None:
        """The generator is a pure function of the registry, so a change to a tool is diffable."""
        again = catalog_page.catalog()

        assert _as_json(again) == _as_json(data), "two runs of the generator disagree"
        assert catalog_page.build_page(again) == catalog_page.build_page(data), "two renders of one catalog disagree"

    def test_page_and_json_are_written_from_a_single_catalog(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any, data: dict[str, Any]
    ) -> None:
        """The model-facing artefact and the human-facing one cannot describe different tools.

        Not by comparing markup against data -- that would mean parsing HTML and pinning layout --
        but by pinning where each comes from: a catalog nobody could have hardcoded is substituted,
        and both files have to be that catalog.
        """
        stub = {"tool_count": 1, "tools": [data["tools"][0]]}
        html_path, json_path = tmp_path / "catalog.html", tmp_path / "catalog.json"
        monkeypatch.setattr(catalog_page, "OUT_DIR", tmp_path)
        monkeypatch.setattr(catalog_page, "HTML_PATH", html_path)
        monkeypatch.setattr(catalog_page, "JSON_PATH", json_path)
        monkeypatch.setattr(catalog_page, "catalog", lambda: stub)

        catalog_page.main()

        assert json.loads(json_path.read_bytes()) == json.loads(_as_json(stub)), "the JSON is not the catalog"
        assert html_path.read_text(encoding="utf-8") == catalog_page.build_page(stub), (
            "the page was not rendered from the catalog that was written as JSON"
        )

    def test_every_catalogued_tool_is_rendered_onto_the_page(self, data: dict[str, Any]) -> None:
        """The one structural claim about the page: a tool present in the data reaches the markup.

        Deliberately the narrowest such claim there is -- a tool's name and its one-line
        description appear somewhere in the page -- because it catches a whole tool dropped between
        the data and the markup, which no other test here can see, and it survives any restyling of
        the page around it. The description is checked as well as the name because a name alone is
        satisfied by an index entry pointing at a tool the page never actually describes.

        Escaping is tolerated rather than pinned: the page may carry the prose verbatim or with its
        punctuation escaped, and either is the tool's own description reaching the reader.
        """
        page = catalog_page.build_page(data)

        missing = [
            tool.name
            for tool in _registered()
            if tool.name not in page
            or not (tool.description in page or html.escape(tool.description, quote=True) in page)
        ]
        assert not missing, f"catalogued but not described on the page: {missing}"
