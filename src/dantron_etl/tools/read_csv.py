"""ReadCsv: a delimited text file becomes one frame, inferred and then corrected column by column.

The catalog's first text source, and the sibling of ``read_excel`` rather than a format switch on
it: a cell is typed and a text file is not, so this tool's whole first act is inference, and its
hazards -- a ragged line, an encoding nobody declared, a blank that is not an empty string -- are
not a workbook's hazards at all.

**Inference reads the whole file and is the default**, corrected per column through
``column_types``: a whole-number column is read whole, a figure column is read as a figure, and a
date stays text because a file cannot say in which order its digits are written. ``column_types``
names only the columns inference gets wrong; there is no ``infer_schema_length``, because sampling
would make a column's type depend on where in the file this month's exception happens to sit.

**A value that will not convert to its declared type stops the run, and so does a line whose field
count disagrees with the header** -- the same tension, the same answer: padding a short line or
turning a bad value into a null both produce a frame that looks right and is not, so both refuse
instead, naming the row or the line rather than guessing at it.

**A blank field is nothing, never an empty string**, and a line of nothing is one blank field like
any other, not layout to be dropped: the paired writer produces exactly that line for a row of
nulls, and dropping it here would lose that row on the way through.
"""

import csv
import datetime as dt
import io
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import polars as pl

from ..column_types import (
    DATE_DEFAULT_FORMAT,
    DATETIME_DEFAULT_FORMAT,
    DECLARED_DTYPE,
    DECLARED_TYPES,
    TEMPORAL_TYPES,
    TIME_DEFAULT_FORMAT,
)
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

# The seven dtype names a declared column may use, spelled the way the query engine already spells
# them in every dtype this catalog prints -- a second vocabulary for the same seven things would be
# the two-dialect mistake this library refuses elsewhere.

# The declared-type vocabulary lives in one place, because more than one tool lets a flow name a
# column's type and two spellings of the same words drift. Bound to module names here so the rest of
# this file reads as it always did; DECLARED_TYPES stays part of this tool's published surface.
_TEMPORAL_TYPES: Final[frozenset[str]] = TEMPORAL_TYPES
_DECLARED_DTYPE: Final[dict[str, pl.DataType]] = DECLARED_DTYPE

_DATE_DEFAULT_FORMAT: Final[str] = DATE_DEFAULT_FORMAT
_DATETIME_DEFAULT_FORMAT: Final[str] = DATETIME_DEFAULT_FORMAT
_TIME_DEFAULT_FORMAT: Final[str] = TIME_DEFAULT_FORMAT

_INT_LITERAL: Final[re.Pattern[str]] = re.compile(r"[-+]?\d+")
_FLOAT_LITERAL: Final[re.Pattern[str]] = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_FRACTION: Final[re.Pattern[str]] = re.compile(r"\.\d+$")
# A number written with a zero in front of another digit, which reading it as a number would drop.
_LEADING_ZERO: Final[str] = r"^[-+]?0\d"
_LEADING_ZERO_PATTERN: Final[re.Pattern[str]] = re.compile(_LEADING_ZERO)
_WHOLE_NUMBER: Final[str] = r"^[-+]?\d+$"

# A byte a spreadsheet writes and almost nothing else expects; accepted and dropped rather than
# left to become part of the first column's name.
_BOM: Final[str] = "﻿"

_QUOTE: Final[str] = '"'

_ACCEPTED_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".tsv")

# Every encoding this tool will decode a file as. A closed set rather than "any string codecs.lookup
# accepts", for the same reason every enum in this library is closed: a template that took anything
# would report a typo as a decoding failure three lines further down instead of at authoring time.
_DECLARED_ENCODINGS: Final[tuple[str, ...]] = (
    "utf8",
    "utf-8",
    "utf-8-sig",
    "latin1",
    "latin-1",
    "iso-8859-1",
    "cp1252",
    "windows-1252",
    "ascii",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
)

_EXAMPLE_PATH: Final[str] = f"{SCRATCH_TOKEN}/rows.csv"


class ReadCsv:
    """Read one delimited text file onto the ``rows`` port, inferred and then corrected."""

    name = "read_csv"
    description = "Read a delimited text file onto the `rows` port, inferring types by column."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The file to read; must end in .csv or .tsv.",
                required=True,
                non_empty=True,
            ),
            "delimiter": ConfigField(
                type="string",
                description="The one character separating fields.",
                default=",",
            ),
            "encoding": ConfigField(
                type="string",
                description="The encoding the file was written in; a fact about the file, not a "
                "choice this tool makes.",
                default="utf8",
                enum=_DECLARED_ENCODINGS,
            ),
            "has_header": ConfigField(
                type="boolean",
                description="Whether the first row (after skip_rows) names the columns.",
                default=True,
            ),
            "skip_rows": ConfigField(
                type="integer",
                description="How many lines above the header (or the data, if has_header is false) to skip.",
                default=0,
            ),
            "column_types": ConfigField(
                type="array",
                description="Columns whose inferred type is wrong, corrected one entry at a time.",
                required=False,
                items=ConfigField(
                    type="object",
                    description="One column's override: which one, what it is, and how it is written.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The header name this entry corrects.",
                            required=True,
                            non_empty=True,
                        ),
                        "type": ConfigField(
                            type="string",
                            description="The declared type, spelled the way the query engine spells it.",
                            required=True,
                            enum=DECLARED_TYPES,
                        ),
                        "format": ConfigField(
                            type="string",
                            description="The order a Date, Datetime or Time is written in; ISO order if left out.",
                            required=False,
                            non_empty=True,
                        ),
                    },
                ),
            ),
        }

    def help(self) -> ToolHelp:
        return ToolHelp(
            detailed_description=(
                "The whole file is read once to infer every column's type; 'column_types' names the "
                "columns that inference gets wrong and corrects only those, one entry at a time. A "
                "blank field arrives as null, never an empty string, and a line whose field count "
                "disagrees with the header stops the run naming the line rather than being padded "
                "or truncated."
            ),
            examples=(_reads_a_file_example(),),
            notes=(
                "There is no infer_schema_length: inference reads the whole file, because a sample "
                "would make a column's type depend on where this month's exception happens to sit.",
                "A value that will not convert to its declared type stops the run naming the column, "
                "the value and the declared type, rather than becoming a null with no other symptom.",
                "A column that would be read as numbers stops the run when a number would change a "
                "value: a leading zero (02134) or a whole number too big for 64 bits. Declare it in "
                "column_types: String keeps it as written, Int64 or Float64 reads it as a number anyway.",
                "The encoding is configurable because it is a fact about a file somebody else wrote; "
                "a byte order mark is accepted and dropped, and bytes invalid in the declared "
                "encoding are refused rather than read lossily.",
            ),
            field_help={
                "column_types": "Each entry names one column and the type it really is; every other "
                "column is inferred as it would have been with no entries at all.",
                "skip_rows": "Lines above the header (or the data) to skip; must not be negative.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong format, a negative skip count, or a column_types entry the file can't fix.

        Every one of these is answerable from the configuration alone, before a file is opened, so
        each is refused here rather than after the whole file has already been read.
        """
        path = config["path"]
        if not path.lower().endswith(_ACCEPTED_EXTENSIONS):
            raise ConfigError(f"read_csv only reads {' or '.join(_ACCEPTED_EXTENSIONS)} files; got path '{path}'")
        if config["skip_rows"] < 0:
            raise ConfigError(f"config field 'skip_rows' must not be negative, got {config['skip_rows']}")
        _validate_column_types(config["column_types"] or [])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a file that is not there, before there is anything else to check.

        A source has no ports to check columns against, so this is the one question this phase can
        still answer for it: is the world this configuration describes actually there?
        """
        path = Path(config["path"])
        if not path.is_file():
            raise ConfigError(f"read_csv found no file at '{config['path']}'")
        if path.stat().st_size == 0:
            raise ConfigError(f"read_csv found no bytes in '{config['path']}'")

    def execute(self, inputs: Ports, config: Mapping[str, Any]) -> Ports:
        path = config["path"]
        delimiter = config["delimiter"]
        encoding = config["encoding"]
        has_header = config["has_header"]
        skip_rows = config["skip_rows"]
        column_types = config["column_types"] or []

        text = _decode(Path(path).read_bytes(), encoding, path)
        text = _strip_bom(text)
        text = _normalise_newlines(text)
        remainder = _skip_lines(text, skip_rows)

        records = _parse_records(remainder, delimiter, skip_rows)
        if not records:
            raise ConfigError(f"read_csv found nothing to read in '{path}' after skipping {skip_rows} row(s)")

        header_fields = records[0][1]
        expected_fields = len(header_fields)
        for line_number, fields in records[1:]:
            if len(fields) != expected_fields:
                raise DataError(
                    f"read_csv refuses line {line_number}: found {len(fields)} field(s), the header "
                    f"has {expected_fields}",
                    offending=pl.DataFrame({"line": [line_number], "fields_found": [len(fields)]}),
                    total=len(records) - 1,
                )

        if has_header:
            _validate_header(header_fields)
            columns = list(header_fields)
            data_records = records[1:]
        else:
            columns = [f"column_{index + 1}" for index in range(expected_fields)]
            data_records = records

        declared_by_column = {entry["column"]: entry for entry in column_types}
        unknown = [name for name in declared_by_column if name not in columns]
        if unknown:
            raise ConfigError(
                f"read_csv declares column_types for column(s) {', '.join(unknown)} not present in the "
                f"header; the header has: {', '.join(columns)}"
            )

        grid: dict[str, list[str | None]] = {name: [] for name in columns}
        for _, fields in data_records:
            for name, raw in zip(columns, fields):
                grid[name].append(raw if raw != "" else None)

        line_numbers = [line_number for line_number, _ in data_records]
        series = [
            _declared_series(name, grid[name], declared_by_column[name])
            if name in declared_by_column
            else _infer_series(name, grid[name], line_numbers)
            for name in columns
        ]

        return Ports({"rows": pl.DataFrame(series)})


def _validate_column_types(column_types: Sequence[Mapping[str, Any]]) -> None:
    """Refuse two rulings for one column, or a format on a type no format says anything about."""
    seen: set[str] = set()
    for entry in column_types:
        column = entry["column"]
        declared_type = entry["type"]
        fmt = entry.get("format")
        if column in seen:
            raise ConfigError(f"read_csv declares column_types for column {column!r} more than once")
        seen.add(column)
        if fmt is not None and declared_type not in _TEMPORAL_TYPES:
            raise ConfigError(
                f"read_csv refuses a format on column {column!r}: format only constrains "
                f"{', '.join(sorted(_TEMPORAL_TYPES))}, not {declared_type!r}"
            )


def _validate_header(fields: Sequence[str]) -> None:
    """Refuse a header that repeats or omits a name, rather than repair it in this reader's own words."""
    seen: set[str] = set()
    for index, value in enumerate(fields, start=1):
        if value == "":
            raise ConfigError(
                f"read_csv refuses a header with a blank heading at position {index}; a header cannot be guessed"
            )
        if value in seen:
            raise ConfigError(
                f"read_csv refuses a header naming {value!r} more than once; a duplicate cannot be "
                "told apart by guessing"
            )
        seen.add(value)


def _decode(raw: bytes, encoding: str, path: str) -> str:
    """Decode ``raw`` strictly as ``encoding``, refusing rather than reading a bad byte lossily."""
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise ConfigError(f"read_csv could not decode '{path}' as {encoding!r}: {error}") from error


def _strip_bom(text: str) -> str:
    return text[1:] if text.startswith(_BOM) else text


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _skip_lines(text: str, skip_rows: int) -> str:
    """``text`` with its first ``skip_rows`` physical lines removed, wherever quoting sits after them."""
    if skip_rows == 0:
        return text
    index = 0
    for _ in range(skip_rows):
        position = text.find("\n", index)
        if position == -1:
            return ""
        index = position + 1
    return text[index:]


def _parse_records(remainder: str, delimiter: str, skip_rows: int) -> list[tuple[int, list[str]]]:
    """Every logical record in ``remainder`` as ``(line number, fields)``, quoting handled correctly.

    The vectorised route when no quote character sits anywhere in the file: a plain split is exact
    once quoting cannot be present to embed the delimiter or a line ending inside a field. The
    quote-aware route otherwise, because only it gets a field spanning two physical lines right.
    """
    if _QUOTE not in remainder:
        return _parse_records_fast(remainder, delimiter, skip_rows)
    return _parse_records_quoted(remainder, delimiter, skip_rows)


def _parse_records_fast(remainder: str, delimiter: str, skip_rows: int) -> list[tuple[int, list[str]]]:
    lines = remainder.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [(skip_rows + offset + 1, line.split(delimiter)) for offset, line in enumerate(lines)]


def _parse_records_quoted(remainder: str, delimiter: str, skip_rows: int) -> list[tuple[int, list[str]]]:
    reader = csv.reader(io.StringIO(remainder), delimiter=delimiter, quotechar=_QUOTE, doublequote=True)
    records: list[tuple[int, list[str]]] = []
    for row in reader:
        fields = row if row else [""]
        records.append((skip_rows + reader.line_num, fields))
    return records


def _infer_series(name: str, values: list[str | None], line_numbers: list[int]) -> pl.Series:
    """``values`` typed as a reader would: a whole number, then a figure, then a truth value, then text.

    A column with nothing but nulls in it is left as text: there is no non-null value to infer a
    type from, and casting an all-null column would otherwise adopt whichever candidate happened to
    be tried first.

    A column that would read as numbers is refused instead when a number would change what the file
    says: a leading zero, or a whole number too big for 64 bits (see ``_refuse_a_lossy_number``).
    """
    series = pl.Series(name, values, dtype=pl.String)
    non_null = series.drop_nulls()
    if non_null.is_empty():
        return series

    for dtype in (pl.Int64, pl.Float64):
        try:
            typed = series.cast(dtype, strict=True)
        except pl.exceptions.PolarsError:
            continue
        _refuse_a_lossy_number(series, line_numbers)
        return typed

    if non_null.is_in(["true", "false"]).all():
        frame = series.to_frame()
        column = pl.col(name)
        mapped = frame.select(pl.when(column == "true").then(True).when(column == "false").then(False).alias(name))
        return mapped[name]

    return series


def _refuse_a_lossy_number(series: pl.Series, line_numbers: list[int]) -> None:
    """Refuse a column of numbers when reading it as numbers would change a value.

    Two cases. A leading zero (``02134``) is dropped by a number. A whole number too big for 64 bits
    can only be read as a figure, which rounds it. The author settles either one in column_types:
    String keeps the value as written, and a number type accepts the change.
    """
    name = series.name
    frame = pl.DataFrame({"line": line_numbers, name: series})
    value = pl.col(name)
    leading_zero = value.str.contains(_LEADING_ZERO)
    too_big = value.str.contains(_WHOLE_NUMBER) & value.cast(pl.Int64, strict=False).is_null()
    offending = frame.filter(leading_zero | too_big)
    if offending.is_empty():
        return
    line, written = offending.row(0)
    if _LEADING_ZERO_PATTERN.match(written):
        why = f"read as a number, {written!r} on line {line} would lose its leading zero"
    else:
        why = f"{written!r} on line {line} is a whole number too big for Int64, and read as a figure it would be rounded"
    raise DataError(
        f"read_csv cannot guess the type of column {name!r}: {why}. Say what the column is in "
        f"column_types: type String keeps every value as written, and a number type reads it as a "
        f"number anyway",
        offending=offending,
        total=series.len(),
    )


def _declared_series(name: str, values: list[str | None], entry: Mapping[str, Any]) -> pl.Series:
    """``values`` converted to the type ``entry`` declares, or a refusal naming what would not convert."""
    declared_type = entry["type"]
    convert = _converter(declared_type, entry.get("format"))

    converted: list[Any] = []
    failures: list[str] = []
    for raw in values:
        if raw is None:
            converted.append(None)
            continue
        try:
            converted.append(convert(raw))
        except ValueError:
            failures.append(raw)

    if failures:
        raise DataError(
            f"read_csv could not convert column {name!r} to declared type {declared_type!r}",
            offending=pl.DataFrame({name: failures}),
            total=len(values),
        )

    return pl.Series(name, converted, dtype=_DECLARED_DTYPE[declared_type])


def _converter(declared_type: str, fmt: str | None) -> Callable[[str], Any]:
    if declared_type == "String":
        return lambda text: text
    if declared_type == "Int64":
        return _parse_int
    if declared_type == "Float64":
        return _parse_float
    if declared_type == "Boolean":
        return _parse_boolean
    if declared_type == "Date":
        resolved = fmt or _DATE_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(text, resolved).date()
    if declared_type == "Datetime":
        resolved = fmt or _DATETIME_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(_truncate_fraction(text, resolved), resolved)
    if declared_type == "Time":
        resolved = fmt or _TIME_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(_truncate_fraction(text, resolved), resolved).time()
    raise AssertionError(f"unreachable declared type {declared_type!r}")  # pragma: no cover


def _truncate_fraction(text: str, fmt: str) -> str:
    """``text`` with a fractional-second part longer than microseconds cut down to it.

    A python ``datetime`` keeps at most six digits of a second; a value written with more --
    nanosecond precision, the resolution this catalog's Time dtype actually carries -- would
    otherwise leave ``strptime`` with unconverted digits and no way to read the rest of the value.
    """
    if "%f" not in fmt:
        return text
    match = _FRACTION.search(text)
    if match is None:
        return text
    return text[: match.start()] + match.group(0)[:7]


def _parse_int(text: str) -> int:
    if not _INT_LITERAL.fullmatch(text):
        raise ValueError(f"{text!r} is not a whole number")
    return int(text)


def _parse_float(text: str) -> float:
    if not _FLOAT_LITERAL.fullmatch(text):
        raise ValueError(f"{text!r} is not a number")
    return float(text)


def _parse_boolean(text: str) -> bool:
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{text!r} is not a boolean literal")


@contextmanager
def _example_file_setup() -> Iterator[None]:
    """A small file the example reads, written here rather than inside ``help()``.

    ``help()`` must describe a tool without changing the process doing the describing -- the
    catalog page generator calls it on every tool -- so the file this example reads is declared
    through ``setup`` instead. Planted with :func:`scratch_path`, into the directory
    :func:`check_tool` opens for the run; nothing here tears it back down, because that directory is
    the checker's to remove, not this setup's.
    """
    target = scratch_path(_EXAMPLE_PATH)
    target.write_bytes(b"key,label\n1,first\n2,second\n")
    yield


def _reads_a_file_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read a two-row file with no columns to correct",
        configuration={"path": _EXAMPLE_PATH},
        explanation="Both columns are inferred from the whole file: 'key' arrives a whole number and "
        "'label' arrives text, with no column_types entry needed for either.",
        inputs=Ports({}),
        outputs=outputs,
        check_row_order=True,
        setup=_example_file_setup,
    )


REGISTRY.register(ReadCsv())
