"""
The roots of the exceptions and warnings raised by xtl.

What a caller can catch:

* ``except XTLCoreError`` (:class:`xtl_core.errors.XTLCoreError`): the exceptions that xtl and xtl-core raise on
  purpose; :class:`XTLError` subclasses it.
* ``except XTLError``: xtl's own exceptions, e.g. :class:`xtl.jobs.exceptions.SubprocessError`.
* Specific exceptions, which also subclass the matching built-in, so catching the built-in works too:

  * :class:`xtl_core.options.exceptions.OptionsValidationError` (a :class:`ValueError`): data that don't validate as
    an :class:`~xtl_core.options.base.Options` model, e.g. a settings file;
  * :class:`xtl.jobs.exceptions.SubprocessError` (a :class:`RuntimeError`): a job's subprocess failed; has the
    ``command`` that was run.

* Everything else is a built-in exception: :class:`TypeError` for an argument of the wrong type, :class:`ValueError`
  for a wrong value, :class:`OSError` and its subclasses for files, and so on.

Warnings are issued as :class:`XTLWarning`.
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
