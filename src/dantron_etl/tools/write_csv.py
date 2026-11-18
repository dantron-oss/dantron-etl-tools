"""WriteCsv: one frame becomes one delimited text file, and the same rows pass through.

The catalog's second file sink, and the sibling of ``write_excel`` rather than a format switch on
it: a text file has no row limit, no cell holding a double, and no tab-naming rules, so none of the
workbook sink's refusals apply here -- and this format has hazards of its own that a cell does not,
which is the other half of why the two are separate tools.

**A whole number is written digit for digit**, and so is a ``Decimal``, because a line of text has
no notion of a double to round either into: the workbook sink refuses both for exactly the reason
this one does not need to.

**A value carrying the delimiter, a quote or a line ending is quoted**, and only that value: the
format's own convention -- wrapped in ``"``, an inner quote doubled -- is what the paired reader
assumes, so it is not configurable here.

**The bytes are UTF-8 with no byte order mark, and the line ending is a line feed**, neither
configurable, because a sink is choosing rather than discovering a fact about somebody else's file --
the deliberate asymmetry with the paired reader's ``encoding`` field.

**A null and an empty string are both written as an empty field**, and the reader hands both back as
nothing; the loss is one-way and is stated here rather than only assumed. **A frame with no rows
writes a header-only file**, because that is a real, empty delivery, not a run that never happened.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import polars as pl

from ..authoring import (
    REGISTRY,
    SCRATCH_TOKEN,
    ConfigError,
    ConfigField,
    Ports,
    ToolHelp,
    WorkedExample,
    scratch_path,
)

_ACCEPTED_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".tsv")
_QUOTE_CHAR: Final[str] = '"'
_FORBIDDEN_DELIMITERS: Final[frozenset[str]] = frozenset({_QUOTE_CHAR, "\n", "\r"})

_PLAIN_PATH: Final[str] = f"{SCRATCH_TOKEN}/plain.csv"
_OVERWRITE_PATH: Final[str] = f"{SCRATCH_TOKEN}/overwrite.csv"


class WriteCsv:
    """Write ``rows`` to one delimited text file, and hand the same rows on."""

    name = "write_csv"
    description = "Write `rows` to a delimited text file, and hand the same rows on."
    category = "output"
    input_ports = ("rows",)
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="Where the file is written; must end in .csv or .tsv.",
                required=True,
                non_empty=True,
            ),
            "delimiter": ConfigField(
                type="string",
                description="The one character separating fields.",
                default=",",
                non_empty=True,
            ),
            "replace": ConfigField(
                type="boolean",
                description="Whether a file already at 'path' may be overwritten; leave it out and "
                "an existing file is refused.",
                default=False,
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "'rows' is written to 'path' as a delimited text file with a header row and every "
                "row of data below it; the same rows leave on the output port unchanged. An "
                "existing file is never overwritten unless 'replace' says so in writing, though "
                "writing true when nothing occupies the path is accepted rather than refused. A "
                "null and an empty string both arrive on the page as an empty field, and the loss is "
                "one-way: the paired reader hands both back as nothing."
            ),
            examples=(_plain_write_example(), _overwrite_example()),
            notes=(
                "The bytes are UTF-8 with no byte order mark, and the line ending is a line feed; "
                "neither is configurable, because a sink is choosing rather than discovering.",
                "A whole number is written digit for digit, and so is a Decimal: a line of text has "
                "no cell to round either one into, which is the reason this is a separate tool from "
                "the workbook sink rather than a format switch on it.",
                "A List, a Struct or a Duration is refused rather than stringified: each has a "
                "plausible-looking wrong answer as a line of text, and none of them is the value it "
                "started as.",
                "A frame with no rows still writes a header-only file, because that is a delivery "
                "that was empty, not a run that never happened.",
            ),
            field_help={
                "replace": "Defaults to false: a path already occupied is refused until an author "
                "says in writing that overwriting it is intended.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format, an unusable delimiter, or an overwrite nobody authorised.

        All three are answerable from the configuration alone, before a file is ever opened, so each
        is refused here rather than partway through a write that would otherwise have to be undone.
        """
        path = config["path"]
        if not path.lower().endswith(_ACCEPTED_EXTENSIONS):
            raise ConfigError(f"write_csv only writes {' or '.join(_ACCEPTED_EXTENSIONS)} files; got path '{path}'")

        _validate_delimiter(config["delimiter"])

        if Path(path).exists() and not config["replace"]:
            raise ConfigError(
                f"write_csv refuses to overwrite the file already at '{path}'; set replace: true "
                "to overwrite it deliberately"
            )

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a column a line of text cannot carry honestly, before a byte is written.

        Answerable from the schema alone, so it happens before the file is opened -- a refused write
        must not leave a half-written file behind.
        """
        frame = inputs["rows"]
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


def _validate_delimiter(delimiter: str) -> None:
    """Refuse a delimiter that is not exactly one character, or that is the quote or a line ending."""
    if len(delimiter) != 1:
        raise ConfigError(f"config field 'delimiter' must be exactly one character, got {delimiter!r}")
    if delimiter in _FORBIDDEN_DELIMITERS:
        raise ConfigError(
            f"config field 'delimiter' may not be {delimiter!r}: it is the quote character or a line ending"
        )


def _write_csv(frame: pl.DataFrame, path: str, delimiter: str) -> None:
    """Write ``frame`` to ``path``, an empty string indistinguishable from null on the way out.

    Polars' own writer quotes an empty string to tell it apart from a null in the same column, which
    is precisely the distinction this format is not entitled to keep -- a text column blanked out by
    an upstream step and one this reader's own paired file never carried a value for must read back
    identically. A null-mapped copy is written instead of the frame that arrived, so the input
    itself is never touched.
    """
    string_columns = [name for name, dtype in frame.schema.items() if dtype == pl.String]
    to_write = frame
    if string_columns:
        to_write = frame.with_columns(
            pl.when(pl.col(name) == "").then(None).otherwise(pl.col(name)).alias(name) for name in string_columns
        )
    to_write.write_csv(path, separator=delimiter)


def _unsupported_dtype_reason(dtype: pl.DataType) -> str | None:
    """Why ``dtype`` cannot land in a line of text honestly, or ``None`` if it can."""
    if isinstance(dtype, pl.List):
        return "a line of text holds one value, not a list; it would arrive as punctuation"
    if isinstance(dtype, pl.Struct):
        return "a line of text holds one value, not a nested record"
    if isinstance(dtype, pl.Duration):
        return "a length of time has no agreed spelling as text; it would arrive as a bare count"
    return None


@contextmanager
def _overwrite_setup() -> Iterator[None]:
    """A file already occupying the path, standing in for last month's extract.

    Planted with :func:`scratch_path`, which resolves the same ``{scratch}`` token the example's own
    configuration holds, into the directory :func:`check_tool` opens for the run. Nothing here tears
    the file back down: the directory is the checker's, and it takes whatever a setup left inside it.
    """
    target = scratch_path(_OVERWRITE_PATH)
    target.write_bytes(b"last month's extract")
    yield


def _plain_write_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="write a new file where nothing exists yet",
        configuration={"path": _PLAIN_PATH},
        explanation="'replace' is left out, and the path is empty, so no permission is needed; the "
        "same two rows leave on the output port unchanged.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
        # Nothing to plant: the directory the checker opens for this run is empty by construction.
    )


def _overwrite_example() -> WorkedExample:
    frame = pl.DataFrame({"key": ["r1", "r2"], "amount": [10.25, -5.5]})
    return WorkedExample(
        scenario="overwrite a file that is already there, on purpose",
        configuration={"path": _OVERWRITE_PATH, "replace": True},
        explanation="A file already occupies the path; 'replace: true' says in writing that "
        "overwriting it is intended, and the same two rows leave on the output port unchanged "
        "either way.",
        inputs=Ports({"rows": frame}),
        outputs=Ports({"rows": frame}),
        check_row_order=True,
        setup=_overwrite_setup,
    )


REGISTRY.register(WriteCsv())
