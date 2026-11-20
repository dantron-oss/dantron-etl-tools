"""The one error this language raises, and how a refusal is worded.

Every refusal quotes the expression it is about, so a log line points at the line of the file. A
check that finds a problem where the expression's text is not in view raises :class:`Refusal`
instead, and the compiler adds the text.
"""


class ExpressionError(ValueError):
    """An expression could not be compiled; the message is written for the person who wrote it.

    A ValueError, because a bad expression is bad input. Each message names the problem, says what to
    write instead where that is clear, and quotes the expression.
    """


class Refusal(Exception):
    """A refusal found where the expression's text is not in view. The compiler adds the text."""

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem


def error(text: str, problem: str) -> ExpressionError:
    """One refusal: what is wrong, and the expression it is wrong in, truncated if long, for a reader of a log."""
    quoted = text if len(text) <= 120 else f"{text[:120]}..."
    return ExpressionError(f"{problem}, in expression: `{quoted}`")
