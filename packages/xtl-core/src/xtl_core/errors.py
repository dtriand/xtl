"""
The root of the exceptions raised by xtl-core.

Exceptions that callers may want to catch specifically subclass :class:`XTLCoreError` and the matching built-in
(e.g. :class:`~xtl_core.options.exceptions.OptionsValidationError` is also a :class:`ValueError`). xtl's own root,
:class:`xtl.exceptions.XTLError`, subclasses :class:`XTLCoreError` too, so ``except XTLCoreError`` catches the
exceptions of both packages.
"""

__all__ = ['XTLCoreError']


class XTLCoreError(Exception):
    """
    Base class for the exceptions raised by xtl-core and xtl.
    """
