"""Checks about files on disk, shared by the tools that read and write them."""

from pathlib import Path
from typing import Final

from ..authoring import SCRATCH_TOKEN, ConfigError

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


def require_output_folder(tool: str, path: str) -> None:
    """Refuse a path whose folder is not there, so the run does not fail at its last step.

    A path still holding the ``{scratch}`` token names the folder the catalog checker opens for a
    worked example. That folder does not exist until the example runs, so it is not checked here.
    """
    if SCRATCH_TOKEN in path:
        return
    folder = Path(path).parent
    if not folder.is_dir():
        raise ConfigError(
            f"{tool} cannot write to '{path}': the folder '{folder}' does not exist. Create the "
            f"folder first, or change config field 'path'"
        )


def refuse_to_overwrite(tool: str, path: str, field: str, allowed: bool) -> None:
    """Refuse a file already at ``path`` unless the step says, in ``field``, that it may be replaced."""
    if Path(path).exists() and not allowed:
        raise ConfigError(
            f"{tool} refuses to overwrite the file already at '{path}'. A flow that runs again "
            f"writes the same file again, so to replace it on every run, add '{field}: true' to "
            f"this step's config"
        )
