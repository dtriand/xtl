"""
Exceptions raised by :mod:`xtl.jobs`.
"""

from typing import Any

from xtl.exceptions import XTLError

__all__ = ['SubprocessError', 'WorkerError', 'DependencyError']


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


class WorkerError(XTLError, RuntimeError):
    """
    What a job in a worker process left behind that couldn't be sent back as it was, because it doesn't survive
    pickling: an error, with its type, text and traceback as text, or a note that the job's data was dropped. Also a
    :class:`RuntimeError`.

    :param message: The original error's text, or what was dropped.
    :param error_type: The original error's type, as ``<module>.<class>``, if it replaces an error.
    :param traceback: The original error's traceback, as text, if it replaces an error.
    """

    def __init__(self, message: str, error_type: str | None = None, traceback: str | None = None):
        self.error_type = error_type
        self.traceback = traceback
        super().__init__(f'{error_type}: {message}' if error_type else message)


class DependencyError(XTLError, RuntimeError):
    """
    A dependency of a job can't be found where the job would run: a module that doesn't load, or an executable that
    isn't there after the dependency is set up. Also a :class:`RuntimeError`.

    :param message: What is missing, and how it was looked for.
    :param dependencies: The names of the dependencies that are missing.
    """

    def __init__(self, message: str, dependencies: tuple[str, ...] = ()):
        self.message = message
        self.dependencies = tuple(dependencies)
        super().__init__(message)

    def __reduce__(self):
        # Picklable with its arguments, so that it travels back from worker processes as it is
        return type(self), (self.message, self.dependencies)
