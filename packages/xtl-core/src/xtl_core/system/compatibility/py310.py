"""
Stand-ins for what Python 3.10 lacks, imported from here on every version: on Python 3.11+, the standard library's.
Delete this module when dropping support for Python 3.10, and import from the standard library instead.

.. code-block:: python

   from xtl_core.system.compatibility.py310 import Self, StrEnum
"""

import sys

__all__ = ['Self', 'StrEnum']


if sys.version_info < (3, 11):
    from enum import Enum

    from typing_extensions import Self

    class StrEnum(str, Enum):
        """
        Stand-in for :class:`enum.StrEnum` (Python 3.11+): members are strings, and ``str()`` and ``format()`` give
        their value, *e.g.* ``'local'`` rather than ``'ComputeSite.LOCAL'``; ``auto()`` gives the lowercase name.
        """

        def __str__(self) -> str:
            return str.__str__(self)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(self, format_spec)

        # NB: Without `self`, as in the standard library
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

else:
    from enum import StrEnum
    from typing import Self
