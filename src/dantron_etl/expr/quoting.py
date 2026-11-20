"""Spelling a value as text of this language.

A caller that writes a value into an expression escapes it here rather than by hand. A refusal
uses the same spelling to quote a literal back the way the author wrote it.
"""


def escape_text(value: str) -> str:
    """``value`` with every quote doubled, ready to sit *inside* a literal this language reads.

    For a caller writing a value into text that already carries the author's own quotes, or the
    language's own ``date'...'`` tag. See :func:`quote_text` for the standalone spelling.
    """
    return value.replace("'", "''")


def quote_text(value: str) -> str:
    """`value` as a complete text literal of this language: quoted, with inner quotes doubled.

    Callers writing a value into an expression should use this, not their own string formatting.
    """
    return f"'{escape_text(value)}'"
