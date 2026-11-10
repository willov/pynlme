"""
Error types for pynlme.
"""


class pynlmeError(Exception):
    """Base exception for pynlme errors."""

    pass


class ConvergenceError(pynlmeError):
    """Raised when algorithm fails to converge."""

    pass


class ModelEvaluationError(pynlmeError):
    """Raised when model function evaluation fails."""

    pass


class ValidationError(pynlmeError):
    """Raised when input validation fails."""

    pass


class WeakIdentificationWarning(UserWarning):
    """Warns that the data barely determine a parameter (see
    :func:`pynlme.nlmefit.nlmefitsa`). ``index`` is the parameter's column of
    ``phi``; ``renamed`` gives the same warning with parameter names."""

    def __init__(self, text: str, index: int):
        super().__init__(text.format(name=f"phi[:, {index}]"))
        self.text, self.index = text, index

    def renamed(self, names) -> "WeakIdentificationWarning":
        out = WeakIdentificationWarning(self.text, self.index)
        out.args = (self.text.format(name=f"'{names[self.index]}'"),)
        return out
