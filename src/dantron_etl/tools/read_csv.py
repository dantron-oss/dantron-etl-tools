"""ReadCsv: read a delimited text file into one frame, working out each column's type.

Types are worked out from the whole file and corrected per column with `column_types`. A value that
will not convert, or a line with the wrong number of fields, stops the run. A blank field is null.
An exact `Decimal` is never worked out; it is read only where `column_types` names it.

Why: docs/design/tools.md#read_csv
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
    EXACT_DIGITS,
    EXACT_TYPE,
    TEMPORAL_TYPES,
    TIME_DEFAULT_FORMAT,
    check_places,
    exact_from_text,
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
from ._files import require_file_to_read, validate_delimiter

_INT_LITERAL: Final[re.Pattern[str]] = re.compile(r"[-+]?\d+")
_FLOAT_LITERAL: Final[re.Pattern[str]] = re.compile(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?")
_FRACTION: Final[re.Pattern[str]] = re.compile(r"\.\d+$")
# A number written with a zero in front of another digit, which reading it as a number would drop.
_LEADING_ZERO: Final[str] = r"^[-+]?0\d"
_LEADING_ZERO_PATTERN: Final[re.Pattern[str]] = re.compile(_LEADING_ZERO)
_WHOLE_NUMBER: Final[str] = r"^[-+]?\d+$"

# A byte order mark, which spreadsheets write; dropped so it does not join the first column's name.
_BOM: Final[str] = "﻿"

_QUOTE: Final[str] = '"'

_ACCEPTED_EXTENSIONS: Final[tuple[str, ...]] = (".csv", ".tsv")

# The encodings accepted. A closed list, so a typo is refused at load, not as a decoding failure.
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
    """Read one delimited text file onto the `rows` port, types worked out then corrected."""

    name = "read_csv"
    description = "Read a CSV or other delimited text file into a table."
    category = "input"
    input_ports = ()
    output_ports = ("rows",)

    def config_template(self) -> Mapping[str, ConfigField]:
        return {
            "path": ConfigField(
                type="string",
                description="The file to read. It must end in .csv or .tsv.",
                required=True,
                non_empty=True,
                file_path=True,
            ),
            "delimiter": ConfigField(
                type="string",
                description="The character between fields. Left out, a comma.",
                default=",",
            ),
            "encoding": ConfigField(
                type="string",
                description="The text encoding the file was saved in. Left out, UTF-8.",
                default="utf8",
                enum=_DECLARED_ENCODINGS,
            ),
            "has_header": ConfigField(
                type="boolean",
                description="Whether the first line (after skip_rows) holds the column names.",
                default=True,
            ),
            "skip_rows": ConfigField(
                type="integer",
                description="How many lines to skip before the header, or before the data without one.",
                default=0,
            ),
            "column_types": ConfigField(
                type="array",
                description="Columns whose worked-out type is wrong, with the type they should be.",
                required=False,
                items=ConfigField(
                    type="object",
                    description="One column: its name, its type, and for dates how it is written.",
                    fields={
                        "column": ConfigField(
                            type="string",
                            description="The column name from the header, such as 'zip'.",
                            required=True,
                            non_empty=True,
                        ),
                        "type": ConfigField(
                            type="string",
                            description="The type, such as 'String', 'Int64', 'Float64', 'Decimal' or 'Date'.",
                            required=True,
                            enum=DECLARED_TYPES,
                        ),
                        "places": ConfigField(
                            type="integer",
                            description="Only with type 'Decimal', and required there: the digits "
                            "after the point, from 0 to 38, such as 2.",
                        ),
                        "format": ConfigField(
                            type="string",
                            description="For a date or time: how it is written, such as '%d/%m/%Y'. "
                            "Left out, year-month-day.",
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
                "Use read_csv to start a flow from a .csv or .tsv file. The whole file is read to work "
                "out each column's type: whole numbers, decimals, true/false, or text. Dates stay text, "
                "because a file cannot say which order its digits are in; name them in 'column_types' "
                "with their format. The rows go to `rows`."
            ),
            examples=(_reads_a_file_example(),),
            notes=(
                "Types are worked out from the whole file, not a sample, so a column's type does not "
                "depend on where an odd value sits.",
                "'column_types' corrects only the columns you list; every other column is read as it "
                "would be without it. A column listed twice is refused, and so is one not in the header.",
                "A value that will not convert to its listed type stops the run, naming the column, the "
                "value and the type. It never becomes a silent null.",
                "A column that would be read as numbers stops the run when a number would change a "
                "value: a leading zero (02134) or a whole number too big for 64 bits.",
                "Settle that in 'column_types': String keeps it as written, Int64 or Float64 reads it "
                "as a number anyway.",
                "A column is read as an exact Decimal only when 'column_types' says so, with its "
                "'places'; figures are otherwise Float64. Each value must be written as plain digits, "
                "with an optional point and minus sign, and fit the places: 1.005 does not fit 2 places "
                "and stops the run. A blank field is null.",
                "A line with more or fewer fields than the header stops the run, naming the line. It is "
                "never padded or cut.",
                "A blank field is null, never empty text. A blank line is a row of nulls, not skipped, "
                "because write_csv writes a row of nulls that way.",
                "A header with a blank or repeated name is refused. Without a header, columns are named "
                "column_1, column_2 and so on.",
                "The encoding is a fact about the file. A byte order mark is accepted and dropped. Bytes "
                "that are invalid in the stated encoding are refused, never read lossily.",
                "An empty file, or one with nothing left after 'skip_rows', is refused.",
            ),
            field_help={
                "path": "For example 'data/orders.csv'. It must end in .csv or .tsv.",
                "delimiter": "For example ';' or a tab. One character.",
                "encoding": "For example 'cp1252' for a file saved by older Windows programs.",
                "column_types": "For example [{column: zip, type: String}, {column: price, type: Decimal, "
                "places: 2}, {column: ordered_on, type: Date, format: '%d/%m/%Y'}].",
                "skip_rows": "For example 2, to skip two title lines above the header. Not negative.",
            },
        )

    def validate_config(self, config: Mapping[str, Any]) -> None:
        """Refuse the wrong extension, a bad delimiter, a negative skip, and a bad column_types list."""
        path = config["path"]
        if not path.lower().endswith(_ACCEPTED_EXTENSIONS):
            raise ConfigError(f"read_csv only reads {' or '.join(_ACCEPTED_EXTENSIONS)} files; got path '{path}'")
        validate_delimiter(config["delimiter"])
        if config["skip_rows"] < 0:
            raise ConfigError(f"config field 'skip_rows' must not be negative, got {config['skip_rows']}")
        _validate_column_types(config["column_types"] or [])

    def validate_inputs(self, inputs: Ports, config: Mapping[str, Any]) -> None:
        """Refuse a file that is missing or empty, before the run starts."""
        require_file_to_read("read_csv", config["path"])
        path = Path(config["path"])
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
    for index, entry in enumerate(column_types):
        column = entry["column"]
        declared_type = entry["type"]
        check_places(declared_type, entry.get("places"), f"column_types[{index}].places")
        fmt = entry.get("format")
        if column in seen:
            raise ConfigError(f"read_csv declares column_types for column {column!r} more than once")
        seen.add(column)
        if fmt is not None and declared_type not in TEMPORAL_TYPES:
            raise ConfigError(
                f"read_csv refuses a format on column {column!r}: format only constrains "
                f"{', '.join(sorted(TEMPORAL_TYPES))}, not {declared_type!r}"
            )


def _validate_header(fields: Sequence[str]) -> None:
    """Refuse a header that repeats or omits a name, rather than repair it in this reader's own words."""
    seen: set[str] = set()
    for index, value in enumerate(fields, start=1):
        if value.strip() == "":
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
    """Every record as `(line number, fields)`, with quoting handled.

    A plain split when the file has no quote character; Python's csv reader otherwise, since only it gets a
    quoted field spanning two lines right.
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
    """`values` typed as a whole number, else a decimal, else true/false, else text.

    An all-null column stays text: there is no value to infer from.
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
    if declared_type == EXACT_TYPE:
        return _exact_series(name, values, entry["places"])
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

    return pl.Series(name, converted, dtype=DECLARED_DTYPE[declared_type])


def _exact_series(name: str, values: list[str | None], places: int) -> pl.Series:
    """``values`` read as exact decimals with ``places``, or a refusal naming the ones that do not fit."""
    text = pl.Series(name, values, dtype=pl.String)
    converted = exact_from_text(text, places)
    if converted.misfits.any():
        raise DataError(
            f"read_csv could not convert column {name!r} to declared type 'Decimal' with {places} places: a "
            f"value converts when written as plain digits, with an optional point and minus sign, such as "
            f"-12.50, with at most {places} places and at most {EXACT_DIGITS - places} digits before the "
            f"point. Correct the file, or declare the column String to keep the text as written",
            offending=pl.DataFrame({name: text.filter(converted.misfits)}),
            total=len(values),
        )
    return converted.values


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
        resolved = fmt or DATE_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(text, resolved).date()
    if declared_type == "Datetime":
        resolved = fmt or DATETIME_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(_truncate_fraction(text, resolved), resolved)
    if declared_type == "Time":
        resolved = fmt or TIME_DEFAULT_FORMAT
        return lambda text: dt.datetime.strptime(_truncate_fraction(text, resolved), resolved).time()
    raise AssertionError(f"unreachable declared type {declared_type!r}")  # pragma: no cover


def _truncate_fraction(text: str, fmt: str) -> str:
    """`text` with fractional seconds cut to six digits, which is all a Python datetime holds."""
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
    """Write the small file the example reads into the example's scratch folder."""
    target = scratch_path(_EXAMPLE_PATH)
    target.write_bytes(b"key,label\n1,first\n2,second\n")
    yield


def _reads_a_file_example() -> WorkedExample:
    outputs = Ports({"rows": pl.DataFrame({"key": [1, 2], "label": ["first", "second"]})})
    return WorkedExample(
        scenario="read a small file",
        configuration={"path": _EXAMPLE_PATH},
        explanation="Each column's type is worked out from the whole file: 'key' is a whole number and 'label' is "
        "text.",
        inputs=Ports({}),
        outputs=outputs,
        check_row_order=True,
        setup=_example_file_setup,
    )


REGISTRY.register(ReadCsv())
