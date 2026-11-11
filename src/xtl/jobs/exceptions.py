"""
Exceptions raised by :mod:`xtl.jobs`.
"""

from typing import Any

from xtl.exceptions import XTLError

__all__ = ['SubprocessError']


class SubprocessError(XTLError, RuntimeError):
    """
    A subprocess run by a job failed, e.g. a batch file that exited with a non-zero return code or couldn't be
    executed. Also a :class:`RuntimeError`.

    :param message: What went wrong.
    :param command: The command that was run, as passed to the subprocess.
    """

    def __init__(self, message: str, command: Any = None):
        self.message = message
        self.command = command
        super().__init__(message)
