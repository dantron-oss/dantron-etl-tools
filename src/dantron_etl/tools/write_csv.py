"""WriteCsv: write one frame to one delimited text file, and pass the same rows on.

UTF-8, line feeds, a header row. A null and empty text are both written as an empty field. An empty
frame stops the run unless `allow_empty`. A file already at `path` needs `overwrite`.

Why: docs/design/tools.md#write_csv
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    SCRATCH_TOKEN,
    ConfigError,
    ConfigField,
    DataError,
    Ports,
    ToolHelp,
    WorkedExample,
    scratch_path,
)
from ._files import refuse_to_overwrite, require_output_folder, validate_delimiter

_ACCEPTED_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".tsv")

_PLAIN_PATH: Final[str] = f"{SCRATCH_TOKEN}/plain.csv"
_OVERWRITE_PATH: Final[str] = f"{SCRATCH_TOKEN}/overwrite.csv"


class WriteCsv:
    """Write `rows` to one delimited text file, and pass the same rows on."""

    name = "write_csv"
    description = "Write a table to a CSV file."
    category = "output"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The file to write. It must end in .csv or .tsv.",
                required=True,
                non_empty=True,
                file_path=True,
            ),
            "delimiter": ConfigField(
                type="string",
                description="The character between fields. Left out, a comma.",
                default=",",
                non_empty=True,
            ),
            "overwrite": ConfigField(
                type="boolean",
                description="Set true to allow replacing a file already at 'path'.",
                default=False,
            ),
            "allow_empty": ConfigField(
                type="boolean",
                description="Set true to write a table with no rows as a file with only the header.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "Use write_csv to deliver a table as a CSV file: a header row, then every row of data. "
                "The same rows also pass on through `rows`, so the flow can continue after it."
            ),
            examples=(_plain_write_example(), _overwrite_example()),
            notes=(
                "A file already at 'path' is refused unless 'overwrite' is true. 'overwrite: true' with "
                "no file there is fine.",
                "The file is UTF-8 with no byte order mark, and lines end with a line feed. Neither can "
                "be changed: unlike reading, writing chooses rather than discovers.",
                "A null and empty text are both written as an empty field, and read_csv reads both back "
                "as null. That difference is lost.",
                "A value containing the delimiter, a quote or a line break is wrapped in quotes, with "
                "inner quotes doubled. Other values are not quoted.",
                "Whole numbers and Decimal values are written digit for digit, a Decimal with every "
                "place it holds (58.899000 at 6 places); text has no limit to round them to. Read "
                "back, the column is Float64 unless read_csv's column_types says Decimal.",
                "A List, Struct or Duration column is refused rather than turned into text.",
                "A table with no rows stops the run unless 'allow_empty' is true, because an empty file "
                "caused by a failure upstream looks like a quiet month.",
                "Allowed, an empty table is written as a header-only file, so the delivery is visibly "
                "empty, not missing.",
                "The folder must already exist. All checks happen before the file is opened, so a "
                "refusal leaves no half-written file.",
            ),
            field_help={
                "path": "For example 'output/orders.csv'.",
                "allow_empty": "Set it when a file may legitimately be empty, such as no refunds this month.",
                "overwrite": "Set it to true for a flow that runs again and again: without it, the "
                "second run is refused because the file is already there.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong extension, a bad delimiter, a missing folder, and an overwrite not allowed."""
        path = config["path"]
        if not path.lower().endswith(_ACCEPTED_EXTENSIONS):
            raise ConfigError(f"write_csv only writes {' or '.join(_ACCEPTED_EXTENSIONS)} files; got path '{path}'")

        validate_delimiter(config["delimiter"])

        require_output_folder("write_csv", path)
        refuse_to_overwrite("write_csv", path, "overwrite", config["overwrite"])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse an empty frame without `allow_empty`, and a column text cannot hold, before writing."""
        frame = inputs["rows"]
        if frame.height == 0 and not config["allow_empty"]:
            raise DataError(
                f"write_csv refuses to write {config['path']!r} from a frame with no rows: an empty file "
                f"looks like a quiet month even when something upstream failed; set allow_empty: true "
                f"if this file may be empty",
                offending=frame,
                total=0,
            )
        for name, dtype in frame.schema.items():
            reason = _unsupported_dtype_reason(dtype)
            if reason is not None:
                raise ConfigError(
                    f"write_csv cannot write column {name!r} of dtype {dtype}: {reason}; format it "
                    "into something a line of text can hold, one step upstream"
                )

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        frame = inputs["rows"]
        _write_csv(frame, config["path"], config["delimiter"])
        return Ports({"rows": frame})


def _write_csv(frame: pl.DataFrame, path: str, delimiter: str) -> None:
    """Write `frame` to `path`, with empty text written exactly like a null.

    polars quotes an empty string to tell it from a null; this format does not keep that difference, so a
    copy with empty text turned to null is written. The input frame is not touched.
    """
    string_columns = [name for name, dtype in frame.schema.items() if dtype == pl.String]
    to_write = frame
    if string_columns:
        to_write = frame.with_columns(
            pl.when(pl.col(name) == "").then(None).otherwise(pl.col(name)).alias(name) for name in string_columns
        )
    to_write.write_csv(path, separator=delimiter)


def _unsupported_dtype_reason(dtype: pl.DataType) -> str | None:
    """Why a column of `dtype` cannot be written as text faithfully, or None if it can."""
    if isinstance(dtype, pl.List):
        return "a line of text holds one value, not a list; it would arrive as punctuation"
    if isinstance(dtype, pl.Struct):
        return "a line of text holds one value, not a nested record"
    if isinstance(dtype, pl.Duration):
        return "a length of time has no agreed spelling as text; it would arrive as a bare count"
    return None


@contextmanager
def _overwrite_setup() -> Iterator[None]:
    """Put a file at the example's path, standing in for last month's extract."""
    target = scratch_path(_OVERWRITE_PATH)
    target.write_bytes(b"last month's extract")
    yield


def _plain_write_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="write a table to a new CSV file",
        configuration={"path": _PLAIN_PATH},
        explanation="No file is at the path yet, so 'overwrite' is not needed. The same two rows pass on through "
        "'rows'.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
    )


def _overwrite_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="replace a file that is already there",
        configuration={"path": _OVERWRITE_PATH, "overwrite": True},
        explanation="A file is already at the path, so 'overwrite: true' is needed to replace it. The same two "
        "rows pass on through 'rows'.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
        setup=_overwrite_setup,
    )


REGISTRY.register(WriteCsv())
