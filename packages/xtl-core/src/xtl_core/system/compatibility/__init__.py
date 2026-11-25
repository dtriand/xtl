"""
Central location for various compatibility checks. Stand-ins for what older Python versions lack are in a module per
version, *e.g.* :mod:`xtl_core.system.compatibility.py310`, which goes once that version is no longer supported.
"""

import sys


#########################
# Python version checks #
#########################
PY_VERS: tuple[int, int, int] = sys.version_info[:3]
"""Version tuple for the current Python interpreter"""

# TODO: Remove when dropping support for Python 3.10 (EOL: 10/2026)
PY310_OR_LESS: bool = PY_VERS < (3, 11)
"""Python version is 3.10 or less"""
# Features missing in Python 3.10 (stand-ins in `py310`):
# - ``enum.StrEnum``
# - ``asyncio.TaskGroup`` (no stand-in)
# - ``from typing import Self``

# TODO: Remove when dropping support for Python 3.11 (EOL: 10/2027)
PY311_OR_LESS: bool = PY_VERS < (3, 12)
"""Python version is 3.11 or less"""

# TODO: Remove when dropping support for Python 3.12 (EOL: 10/2028)
PY312_OR_LESS: bool = PY_VERS < (3, 13)
"""Python version is 3.12 or less"""

# TODO: Remove when dropping support for Python 3.13 (EOL: 10/2029)
PY313_OR_LESS: bool = PY_VERS < (3, 14)
"""Python version is 3.13 or less"""
# Features missing in Python 3.13:
# - Deferred evaluation of annotations (`from __future__ import annotations`)

# TODO: Remove when dropping support for Python 3.14 (EOL: 10/2030)
PY314_OR_LESS: bool = PY_VERS < (3, 15)
"""Python version is 3.14 or less"""
# Features missing in Python 3.14 (stand-ins in `py314`):
# - built-in ``sentinel``


#############
# OS checks #
#############
OS_WINDOWS: bool = sys.platform == 'win32'
"""Operating system is Windows"""

OS_LINUX: bool = sys.platform == 'linux'
"""Operating system is Linux"""

OS_MACOS: bool = sys.platform == 'darwin'
"""Operating system is macOS"""

OS_POSIX: bool = OS_LINUX or OS_MACOS
"""Operating system is POSIX compliant (Linux or macOS)"""
