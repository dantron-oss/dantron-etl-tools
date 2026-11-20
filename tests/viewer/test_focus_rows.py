"""Following one entity through a saved run: PortStore's match argument and focus_match."""

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dantron_etl.focus import FocusSpec, RelatedKey
from dantron_etl.preview import PortStore, PreviewError, persist_ports
from dantron_etl.spec import PortRef

LINES = pl.DataFrame(
    {
        "party": ["P1", "P2", "P1", "P3", "P2", "P1"],
        "bill": ["I1", "I2", "I3", "I4", "I5", "I1"],
        "amount": [10, 20, 30, 40, 20, 60],
        "day": [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 5), date(2026, 2, 1), date(2026, 2, 2), date(2026, 3, 3)],
    }
)
BILLS = pl.DataFrame({"bill": ["I1", "I2", "I3", "I4", "I5"], "total": [70, 20, 30, 40, 20]})
SUMMARY = pl.DataFrame({"month": ["2026-01", "2026-02"], "total": [60, 60]})
NUMBERED = pl.DataFrame({"party_id": [101, 102, 101], "amount": [1, 2, 3]})


@pytest.fixture
def store(tmp_path: Path) -> PortStore:
    persist_ports(
        {"lines.rows": LINES, "bills.rows": BILLS, "summary.rows": SUMMARY, "numbered.rows": NUMBERED},
        tmp_path,
    )
    return PortStore(tmp_path)


def column(store: PortStore, port: str, name: str, **kwargs: object) -> list[str | None]:
    found = store.window(port, limit=500, **kwargs)  # type: ignore[arg-type]
    at = found.columns.index(name)
    return [row[at] for row in found.rows]


class TestWindowMatch:
    def test_keeps_rows_whose_text_is_one_of_the_values(self, store: PortStore) -> None:
        found = store.window("lines.rows", match={"party": ["P1", "P3"]})
        assert found.matched == 4
        assert found.total == 6
        assert {row[0] for row in found.rows} == {"P1", "P3"}

    def test_numeric_and_date_columns_match_on_their_text(self, store: PortStore) -> None:
        assert column(store, "lines.rows", "bill", match={"amount": ["20"]}) == ["I2", "I5"]
        assert column(store, "lines.rows", "bill", match={"day": ["2026-01-05"]}) == ["I1", "I3"]

    def test_match_is_exact_not_a_substring(self, store: PortStore) -> None:
        assert store.window("lines.rows", match={"party": ["P"]}).matched == 0

    def test_a_column_the_port_does_not_carry_is_refused(self, store: PortStore) -> None:
        with pytest.raises(PreviewError, match="nope"):
            store.window("lines.rows", match={"nope": ["x"]})

    def test_combined_with_a_filter_gives_the_intersection(self, store: PortStore) -> None:
        found = store.window("lines.rows", match={"party": ["P1"]}, filters={"day": "2026-01"})
        assert found.matched == 2
        assert column(store, "lines.rows", "amount", match={"party": ["P1"]}, filters={"day": "2026-01"}) == ["10", "30"]

    def test_no_match_leaves_the_window_as_it_was(self, store: PortStore) -> None:
        assert store.window("lines.rows", match=None, offset=1, limit=3) == store.window("lines.rows", offset=1, limit=3)
        assert store.window("lines.rows", match=None, filters={"party": "P2"}) == store.window(
            "lines.rows", filters={"party": "P2"}
        )


def focus_match(*args: object) -> tuple[str, Any]:
    """The function under test, imported per call so the window tests run without it."""
    from dantron_etl.viewer.focus_rows import focus_match as found

    return found(*args)


class TestFocusMatch:
    def test_a_port_with_an_identity_column_matches_on_it(self, store: PortStore) -> None:
        focus = FocusSpec(label="party", identity=("party",))
        kind, match = focus_match(store, focus, "lines.rows", "P1")
        assert kind == "match"
        assert {name: list(values) for name, values in match.items()} == {"party": ["P1"]}
        assert column(store, "lines.rows", "bill", match=match) == ["I1", "I3", "I1"]

    def test_the_first_identity_column_the_port_carries_is_used(self, store: PortStore) -> None:
        focus = FocusSpec(label="party", identity=("party_id", "party"))
        kind, match = focus_match(store, focus, "lines.rows", "P2")
        assert (kind, {name: list(values) for name, values in match.items()}) == ("match", {"party": ["P2"]})
        kind, match = focus_match(store, focus, "numbered.rows", "101")
        assert kind == "match"
        assert column(store, "numbered.rows", "amount", match=match) == ["1", "3"]

    def test_a_port_with_no_identity_column_has_no_grain(self, store: PortStore) -> None:
        focus = FocusSpec(label="party", identity=("party",))
        kind, reason = focus_match(store, focus, "summary.rows", "P1")
        assert kind == "no_grain"
        assert isinstance(reason, str) and reason.strip()

    def test_a_via_port_learns_its_key_values_from_the_origin_port(self, store: PortStore) -> None:
        focus = FocusSpec(
            label="party",
            identity=("party",),
            via={PortRef("bills", "rows"): RelatedKey(PortRef("lines", "rows"), "bill")},
        )
        kind, match = focus_match(store, focus, "bills.rows", "P1")
        assert kind == "match"
        assert set(match) == {"bill"}
        assert sorted(match["bill"]) == ["I1", "I3"]
        assert column(store, "bills.rows", "total", match=match) == ["70", "30"]

    def test_a_value_held_nowhere_matches_zero_rows_without_an_error(self, store: PortStore) -> None:
        focus = FocusSpec(
            label="party",
            identity=("party",),
            via={PortRef("bills", "rows"): RelatedKey(PortRef("lines", "rows"), "bill")},
        )
        kind, match = focus_match(store, focus, "lines.rows", "nobody")
        assert kind == "match"
        assert store.window("lines.rows", match=match).matched == 0

        kind, learned = focus_match(store, focus, "bills.rows", "nobody")
        assert kind == "match"
        assert store.window("bills.rows", match=learned).matched == 0
