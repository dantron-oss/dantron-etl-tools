"""The one error this language raises, and how a refusal is worded.

Every refusal quotes the expression it is about, so a log line points at the line of the file. A
check that finds a problem where the expression's text is not in view raises :class:`Refusal`
instead, and the compiler adds the text.
"""


class ExpressionError(ValueError):
    """An expression could not be compiled, with a message written for whoever wrote the text.

    A ``ValueError`` subclass because a bad expression is bad input, and because a caller that
    already handles configuration errors as ``ValueError`` keeps working without knowing this type
    exists. Every message names the problem, says what to write instead where there is an obvious
    answer, and quotes the offending expression so a log line points at the line of the file.
    """


class Refusal(Exception):
    """A refusal found where the expression's text is not in view. The compiler adds the text."""

    def __init__(self, problem: str) -> None:
        super().__init__(problem)
        self.problem = problem


def error(text: str, problem: str) -> ExpressionError:
    """One refusal: what is wrong, and the expression it is wrong in.

    The expression is quoted in every message because these are read in a log, hours later, by
    someone holding a file of many expressions and needing to know which line to open. Long text is
    truncated so that a pathological input cannot turn one refusal into a page of log.
    """
    quoted = text if len(text) <= 120 else f"{text[:120]}..."
    return ExpressionError(f"{problem}, in expression: `{quoted}`")
