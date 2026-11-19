"""What a caller hands a run: cached frames, supplied sources, and the ports to keep.

Every mistake in these is refused before the first step runs, so no step has written anything yet.
A cached frame carries its port's own fingerprint, so a split's two sides cannot be swapped. A
source is supplied whole or not at all. ``keep`` holds on to the ports a caller will read after a
``retain="live"`` run.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl import CachedFrame, PortRef, RunError, load_pipeline, run_pipeline
from dantron_etl.authoring import ConfigField, Ports, ToolHelp, ToolRegistry, WorkedExample
from dantron_etl.registry import REGISTRY


def _split_document(tmp_path: Path) -> dict[str, Any]:
    """A source, a split, a sort on each side, and a file written from the source first."""
    return {
        "name": "split",
        "steps": [
            {"id": "rows", "tool": "read_csv", "config": {"path": "unused.csv"}},
            {
                "id": "early",
                "tool": "write_csv",
                "config": {"path": str(tmp_path / "early.csv"), "overwrite": True},
                "inputs": {"rows": "rows.rows"},
            },
            {"id": "split", "tool": "filter", "config": {"predicate": "[n] > 1"}, "inputs": {"rows": "rows.rows"}},
            {
                "id": "high",
                "tool": "sort",
                "config": {"by": [{"column": "n"}]},
                "inputs": {"rows": "split.true"},
            },
            {
                "id": "low",
                "tool": "sort",
                "config": {"by": [{"column": "n"}]},
                "inputs": {"rows": "split.false"},
            },
            {"id": "high_ok", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "high.rows"}},
            {"id": "low_ok", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "low.rows"}},
        ],
    }


ROWS = pl.DataFrame({"n": [1, 2, 3]})


class TestACachedFrameBelongsToOnePort:
    def test_the_two_sides_of_a_split_have_different_fingerprints(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        assert spec.fingerprint(PortRef("split", "true")) != spec.fingerprint(PortRef("split", "false"))
        assert spec.fingerprint("split.true") == spec.fingerprint(PortRef("split", "true"))

    def test_a_swapped_cache_is_refused(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        true_frame, false_frame = pl.DataFrame({"n": [2, 3]}), pl.DataFrame({"n": [1]})
        swapped = {
            PortRef("split", "true"): CachedFrame(false_frame, spec.fingerprint(PortRef("split", "false"))),
            PortRef("split", "false"): CachedFrame(true_frame, spec.fingerprint(PortRef("split", "true"))),
        }
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"rows.rows": ROWS}, cached=swapped)
        assert "split.true" in str(raised.value) or "split.false" in str(raised.value)

    def test_the_right_way_round_is_accepted(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        cached = {
            PortRef("split", side): CachedFrame(frame, spec.fingerprint(PortRef("split", side)))
            for side, frame in (("true", pl.DataFrame({"n": [2, 3]})), ("false", pl.DataFrame({"n": [1]})))
        }
        trace = run_pipeline(spec, sources={"rows.rows": ROWS}, cached=cached)
        assert trace.frame("high").to_series().to_list() == [2, 3]

    def test_a_step_fingerprint_on_a_step_with_two_ports_is_refused_saying_what_to_take(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        cached = {
            PortRef("split", side): CachedFrame(pl.DataFrame({"n": [1]}), spec.fingerprint("split"))
            for side in ("true", "false")
        }
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"rows.rows": ROWS}, cached=cached)
        assert "spec.fingerprint(PortRef(" in str(raised.value)

    def test_a_step_with_one_port_has_one_fingerprint_either_way(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        assert spec.fingerprint("high") == spec.fingerprint(PortRef("high", "rows"))
        trace = run_pipeline(
            spec,
            sources={"rows.rows": ROWS},
            cached={"high.rows": CachedFrame(pl.DataFrame({"n": [9]}), spec.fingerprint("high"))},
            until="high_ok",
        )
        assert trace.frame("high").to_series().to_list() == [9]


class _TwoPortSource:
    """A source with two output ports, for the partial-supply refusal."""

    name = "two_port_source"
    description = "Reads nothing and hands back two frames."
    category = "input"
    input_ports: Sequence[str] = ()
    output_ports: Sequence[str] = ("good", "bad")

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:
        frame = pl.DataFrame({"n": [1]})
        return ToolHelp(
            detailed_description="Two frames.",
            examples=(WorkedExample("two", {}, "two", Ports({}), Ports({"good": frame, "bad": frame})),),
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = pl.DataFrame({"n": [1]})
        return Ports({"good": frame, "bad": frame})


def _registry_with_two_port_source() -> ToolRegistry:
    registry = ToolRegistry()
    for name in REGISTRY:
        registry.register(REGISTRY.get(name))
    registry.register(_TwoPortSource())
    return registry


class TestASourceIsSuppliedWholeBeforeAnythingRuns:
    def _document(self, tmp_path: Path) -> dict[str, Any]:
        return {
            "name": "partial",
            "steps": [
                {"id": "first", "tool": "read_csv", "config": {"path": "unused.csv"}},
                {
                    "id": "written",
                    "tool": "write_csv",
                    "config": {"path": str(tmp_path / "written.csv"), "overwrite": True},
                    "inputs": {"rows": "first.rows"},
                },
                {"id": "pair", "tool": "two_port_source", "config": {}},
                {"id": "good_ok", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "pair.good"}},
                {"id": "bad_ok", "tool": "assert_rows", "config": {"predicate": "[n] > 0"}, "inputs": {"rows": "pair.bad"}},
            ],
        }

    def test_a_partly_supplied_source_is_refused_before_any_file_is_written(self, tmp_path: Path) -> None:
        spec = load_pipeline(self._document(tmp_path), registry=_registry_with_two_port_source())
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"first.rows": ROWS, "pair.good": ROWS})
        assert "pair.bad" in str(raised.value)
        assert not (tmp_path / "written.csv").exists()

    def test_the_mirror_with_the_other_port_supplied(self, tmp_path: Path) -> None:
        spec = load_pipeline(self._document(tmp_path), registry=_registry_with_two_port_source())
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"first.rows": ROWS, "pair.bad": ROWS})
        assert "pair.good" in str(raised.value)

    def test_both_ports_supplied_runs(self, tmp_path: Path) -> None:
        spec = load_pipeline(self._document(tmp_path), registry=_registry_with_two_port_source())
        trace = run_pipeline(spec, sources={"first.rows": ROWS, "pair.good": ROWS, "pair.bad": ROWS})
        assert trace.event("pair").supplied

    @pytest.mark.parametrize("value", [ROWS.lazy(), {"n": [1]}, None], ids=["lazy", "dict", "none"])
    def test_a_value_that_is_not_a_dataframe_is_refused_before_anything_runs(self, tmp_path: Path, value: Any) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"rows.rows": value})
        assert "rows.rows" in str(raised.value) and "DataFrame" in str(raised.value)
        assert not (tmp_path / "early.csv").exists()

    def test_a_lazy_frame_is_told_to_collect(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        with pytest.raises(RunError) as raised:
            run_pipeline(spec, sources={"rows.rows": ROWS.lazy()})
        assert ".collect()" in str(raised.value)


class TestKeepHoldsTheNamedPortsUnderALiveRun:
    def test_live_without_keep_holds_nothing_at_the_end(self, tmp_path: Path) -> None:
        trace = run_pipeline(load_pipeline(_split_document(tmp_path)), sources={"rows.rows": ROWS}, retain="live")
        assert dict(trace.frames) == {}

    @pytest.mark.parametrize("named", [PortRef("high", "rows"), "high.rows", "high"])
    def test_a_kept_port_survives_and_nothing_else_does(self, tmp_path: Path, named: Any) -> None:
        trace = run_pipeline(
            load_pipeline(_split_document(tmp_path)), sources={"rows.rows": ROWS}, retain="live", keep=[named]
        )
        assert set(trace.frames) == {PortRef("high", "rows")}
        assert trace.frame("high").to_series().to_list() == [2, 3]

    def test_a_step_named_alone_keeps_every_one_of_its_ports(self, tmp_path: Path) -> None:
        trace = run_pipeline(
            load_pipeline(_split_document(tmp_path)), sources={"rows.rows": ROWS}, retain="live", keep=["split"]
        )
        assert set(trace.frames) == {PortRef("split", "true"), PortRef("split", "false")}

    def test_keep_and_until_together_keep_the_port_of_the_step_asked_for(self, tmp_path: Path) -> None:
        trace = run_pipeline(
            load_pipeline(_split_document(tmp_path)),
            sources={"rows.rows": ROWS},
            retain="live",
            until="low",
            keep=["low.rows"],
        )
        assert set(trace.frames) == {PortRef("low", "rows")}

    def test_the_answers_are_the_same_as_without_keep(self, tmp_path: Path) -> None:
        spec = load_pipeline(_split_document(tmp_path))
        plain = run_pipeline(spec, sources={"rows.rows": ROWS}, retain="live")
        kept = run_pipeline(spec, sources={"rows.rows": ROWS}, retain="live", keep=["high"])
        assert [event.outputs for event in plain.events] == [event.outputs for event in kept.events]

    @pytest.mark.parametrize("named", ["nowhere", "high.nope", "nowhere.rows"])
    def test_a_name_the_flow_does_not_have_is_refused_before_anything_runs(self, tmp_path: Path, named: str) -> None:
        with pytest.raises(RunError) as raised:
            run_pipeline(
                load_pipeline(_split_document(tmp_path)), sources={"rows.rows": ROWS}, retain="live", keep=[named]
            )
        assert "keep" in str(raised.value)
        assert not (tmp_path / "early.csv").exists()
