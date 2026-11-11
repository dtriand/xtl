"""
The roots of the exceptions and warnings raised by xtl.

:class:`XTLError` subclasses :class:`xtl_core.errors.XTLCoreError`, so ``except XTLCoreError`` catches the exceptions
of both xtl and xtl-core; ``except XTLError`` catches xtl's own.
"""

__all__ = [
    'ConfigWarning',
    'DimensionalityError',
    'FileError',
    'MissingArgument',
    'XTLError',
    'XTLWarning',
]

from xtl_core.errors import XTLCoreError

from .base import FileError, DimensionalityError, MissingArgument
from .warnings import ConfigWarning


class XTLError(XTLCoreError):
    """Base class for the exceptions raised by xtl."""


class XTLWarning(UserWarning):
    """Base class for the warnings issued by xtl."""
