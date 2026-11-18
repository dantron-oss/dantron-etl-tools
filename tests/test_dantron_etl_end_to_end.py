"""One realistic flow, run end to end from its YAML file, checked against figures worked out by hand.

``tests/fixtures/end_to_end/monthly_by_band.yaml`` reads two CSVs, types the columns, keeps the
revenue rows, joins each to its party with ``how: left`` (null keys on both sides), bands it with
``case_when``, computes a monthly figure, totals it per band, ranks the bands and writes a CSV.
That is the shape the real flows repeat hundreds of times.

The expected frames are written out below by hand from the input CSVs. Each comparison checks the
schema first, then the whole frame, so a changed type, a lost row or a moved figure all fail.
The flow runs in a scratch folder, so its output never lands in the checkout.
"""

import shutil
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dantron_etl.pipeline import PortRef, read_pipeline_file
from dantron_etl.runner import PipelineTrace, run_pipeline

FLOW_DIR = Path(__file__).resolve().parent / "fixtures" / "end_to_end"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[PipelineTrace, Path]:
    """The flow, run once from a scratch folder holding its two input files."""
    folder = tmp_path_factory.mktemp("end_to_end")
    for name in ("entries.csv", "parties.csv"):
        shutil.copyfile(FLOW_DIR / name, folder / name)
    spec = read_pipeline_file(FLOW_DIR / "monthly_by_band.yaml")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        trace = run_pipeline(spec, retain="all")
    return trace, folder


def _same(actual: pl.DataFrame, expected: pl.DataFrame) -> None:
    """Schema first, so a type change reads as one, then every row and value in order."""
    assert actual.schema == expected.schema
    assert_frame_equal(actual, expected)


_LEFT_SCHEMA = {
    "entry_id": pl.Int64,
    "party_code": pl.String,
    "kind": pl.String,
    "amount": pl.Float64,
    "months": pl.Int64,
}


def test_the_left_join_keeps_every_revenue_row_once_and_pairs_no_null_key(run: tuple[PipelineTrace, Path]) -> None:
    """Entry 6 (no code) and entry 7 (code X, no such party) survive with nothing from the right.

    The parties file also has a row with no code. A join that paired null keys would give entry 6
    the name "Nobody" and the tier "bronze".
    """
    trace, _ = run
    _same(
        trace.frames[PortRef("with_party", "matched")],
        pl.DataFrame(
            {
                "entry_id": [1, 2, 3, 5, 6, 7],
                "party_code": ["N", "N", "B", "C", None, "X"],
                "kind": ["REVENUE"] * 6,
                "amount": [1200.0, 600.0, 300.0, 240.0, 500.0, 90.0],
                "months": [12, 6, 3, 12, 5, 3],
                "party_name": ["Northwind", "Northwind", "Borealis", "Cirrus", None, None],
                "tier": ["gold", "gold", "silver", None, None, None],
            },
            schema={**_LEFT_SCHEMA, "party_name": pl.String, "tier": pl.String},
        ),
    )
    _same(
        trace.frames[PortRef("with_party", "left_only")],
        pl.DataFrame(
            {"entry_id": [6, 7], "party_code": [None, "X"], "kind": ["REVENUE"] * 2, "amount": [500.0, 90.0], "months": [5, 3]},
            schema=_LEFT_SCHEMA,
        ),
    )


def test_the_rows_the_filter_set_aside_are_the_refund_and_the_row_with_no_kind(run: tuple[PipelineTrace, Path]) -> None:
    trace, _ = run
    _same(
        trace.frames[PortRef("revenue", "false")],
        pl.DataFrame(
            {"entry_id": [4, 8], "party_code": ["B", "C"], "kind": ["REFUND", None], "amount": [-100.0, 50.0], "months": [1, 1]},
            schema=_LEFT_SCHEMA,
        ),
    )


def test_each_row_gets_its_band_and_its_monthly_figure(run: tuple[PipelineTrace, Path]) -> None:
    """Gold is top, silver is middle, and anything else, a missing tier included, is other."""
    trace, _ = run
    monthly = trace.frames[PortRef("monthly", "rows")]
    _same(
        monthly.select("entry_id", "band", "monthly"),
        pl.DataFrame(
            {
                "entry_id": [1, 2, 3, 5, 6, 7],
                "band": ["top", "top", "middle", "other", "other", "other"],
                "monthly": [100.0, 100.0, 100.0, 20.0, 100.0, 30.0],
            },
            schema={"entry_id": pl.Int64, "band": pl.String, "monthly": pl.Float64},
        ),
    )


_TOTALS = pl.DataFrame(
    {"band": ["top", "other", "middle"], "monthly_total": [200.0, 150.0, 100.0], "entries": [2, 3, 1]},
    schema={"band": pl.String, "monthly_total": pl.Float64, "entries": pl.Int64},
)


def test_the_published_totals_are_the_hand_worked_figures(run: tuple[PipelineTrace, Path]) -> None:
    """top = 1200/12 + 600/6; other = 240/12 + 500/5 + 90/3; middle = 300/3. Ranked largest first."""
    trace, _ = run
    _same(trace.frames[PortRef("ranked", "rows")], _TOTALS)


def test_the_file_written_holds_the_same_figures(run: tuple[PipelineTrace, Path]) -> None:
    """The CSV on disk, read back with the expected types, is the ranked frame."""
    _, folder = run
    written = pl.read_csv(folder / "totals.csv", schema=_TOTALS.schema)
    _same(written, _TOTALS)
