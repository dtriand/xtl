"""
``xtl.core`` is :mod:`xtl_core`: ``from xtl.core.options import Options`` gives the same
objects as ``from xtl_core.options import Options``. Every module is loaded once, under
its ``xtl_core`` name.
"""

from typing import TYPE_CHECKING

from xtl_core._alias import alias_package

if TYPE_CHECKING:
    # Static tools can't follow the alias
    from xtl_core import *  # noqa: F403

# Replaces this module with `xtl_core` in `sys.modules`
alias_package(__name__, 'xtl_core')
