"""
The roots of the exceptions and warnings raised by xtl.

:class:`XTLError` subclasses :class:`xtl_core.errors.XTLCoreError`, so ``except XTLCoreError`` catches the exceptions
of both xtl and xtl-core; ``except XTLError`` catches xtl's own.
"""

__all__ = [
    'XTLError',
    'XTLWarning',
]

from xtl_core.errors import XTLCoreError


class XTLError(XTLCoreError):
    """Base class for the exceptions raised by xtl."""


class XTLWarning(UserWarning):
    """
    The category of the warnings issued by xtl. To silence them:
    ``warnings.filterwarnings('ignore', category=XTLWarning)``.
    """
