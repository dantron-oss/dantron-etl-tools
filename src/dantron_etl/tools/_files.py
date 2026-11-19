"""Checks about files on disk, shared by the tools that read and write them.

One copy of each check, so the readers ask the same questions in the same words, and so do the
writers. The leading underscore keeps this module out of the tool catalog.
"""

from pathlib import Path
from typing import Final

from ..authoring import ConfigError

_QUOTE_CHAR: Final[str] = '"'
_FORBIDDEN_DELIMITERS: Final[frozenset[str]] = frozenset({_QUOTE_CHAR, "\n", "\r"})


def validate_delimiter(delimiter: str) -> None:
    """Refuse a delimiter that is not exactly one character, or that is the quote or a line ending."""
    if len(delimiter) != 1:
        raise ConfigError(f"config field 'delimiter' must be exactly one character, got {delimiter!r}")
    if delimiter in _FORBIDDEN_DELIMITERS:
        raise ConfigError(
            f"config field 'delimiter' may not be {delimiter!r}: it is the quote character or a line ending"
        )


def require_file_to_read(tool: str, path: str) -> None:
    """Refuse a path with no file at it, naming the path."""
    if not Path(path).is_file():
        raise ConfigError(f"{tool} found no file at '{path}'")
