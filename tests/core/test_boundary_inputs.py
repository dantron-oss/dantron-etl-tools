"""A tool cannot change the frames it was given.

The boundary hands each tool its own copy of each input frame. A change made in place to the copy
never reaches the caller's frame, so another branch reading the same port sees what was produced.
A change to the copy's columns or its number of rows is refused, naming the tool and the port.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import polars as pl
import pytest

from dantron_etl.authoring import ConfigField, PortError, Ports, ToolHelp, run_tool


class _Slip:
    """A tool whose execute does ``slip`` to its input and then passes a fresh frame on."""

    name = "slip"
    description = "Does something to its input."
    category = "transform"
    input_ports: Sequence[str] = ("rows",)
    output_ports: Sequence[str] = ("rows",)

    def __init__(self, slip: Callable[[pl.DataFrame], None]) -> None:
        self._slip = slip

    def config_template(self) -> Mapping[str, ConfigField]:
        return {}

    def help(self) -> ToolHelp:  # pragma: no cover - never registered
        raise NotImplementedError

    def validate_config(self, config: Mapping[str, Any]) -> None:
        return None

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        return None

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        self._slip(inputs["rows"])
        return Ports({"rows": pl.DataFrame({"n": [0]})})


def _set_value(frame: pl.DataFrame) -> None:
    frame[0, "n"] = 999


def _append(frame: pl.DataFrame) -> None:
    frame.extend(pl.DataFrame({"n": [4]}))


def _add_column(frame: pl.DataFrame) -> None:
    frame.insert_column(0, pl.Series("m", [0, 0, 0]))


def _drop_column(frame: pl.DataFrame) -> None:
    frame.drop_in_place("n")


@pytest.mark.parametrize("slip", [_set_value, _append, _add_column, _drop_column])
def test_the_callers_frame_is_never_changed(slip: Callable[[pl.DataFrame], None]) -> None:
    frame = pl.DataFrame({"n": [1, 2, 3]})
    try:
        run_tool(_Slip(slip), Ports({"rows": frame}), {})
    except PortError:
        pass
    assert frame.equals(pl.DataFrame({"n": [1, 2, 3]}))


@pytest.mark.parametrize(
    ("slip", "said"),
    [(_append, "number of rows"), (_add_column, "columns"), (_drop_column, "columns")],
)
def test_a_change_to_the_rows_or_columns_is_refused_by_name(slip: Callable[[pl.DataFrame], None], said: str) -> None:
    with pytest.raises(PortError) as raised:
        run_tool(_Slip(slip), Ports({"rows": pl.DataFrame({"n": [1, 2, 3]})}), {})
    assert "'slip'" in str(raised.value) and "'rows'" in str(raised.value) and said in str(raised.value)


def test_a_value_set_in_place_reaches_nothing_and_is_not_refused() -> None:
    frame = pl.DataFrame({"n": [1, 2, 3]})
    result = run_tool(_Slip(_set_value), Ports({"rows": frame}), {})
    assert result["rows"].to_series().to_list() == [0]
    assert frame["n"].to_list() == [1, 2, 3]


def test_an_empty_input_is_copied_too() -> None:
    frame = pl.DataFrame({"n": []}, schema={"n": pl.Int64})
    with pytest.raises(PortError):
        run_tool(_Slip(_append), Ports({"rows": frame}), {})
    assert frame.height == 0
