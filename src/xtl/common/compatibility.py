"""
Central location for various compatibility checks.
"""
import platform
import sys

#########################
# Python version checks #
#########################
PY_VERS: tuple[int, int, int] = sys.version_info[:3]
"""Version tuple for the current Python interpreter"""

# TODO: Remove when dropping support for Python 3.10 (EOL: 10/2026)
PY310_OR_LESS: bool = (PY_VERS < (3, 11))
"""Python version is 3.10 or less"""
# Features missing in Python 3.10:
# - ``enum.StrEnum`` (available in Python 3.11+)

# TODO: Remove when dropping support for Python 3.11 (EOL: 10/2027)
PY311_OR_LESS: bool = (PY_VERS < (3, 12))
"""Python version is 3.11 or less"""

# TODO: Remove when dropping support for Python 3.12 (EOL: 10/2028)
PY312_OR_LESS: bool = (PY_VERS < (3, 13))
"""Python version is 3.12 or less"""

# TODO: Remove when dropping support for Python 3.13 (EOL: 10/2029)
PY313_OR_LESS: bool = (PY_VERS < (3, 14))
"""Python version is 3.13 or less"""

# TODO: Remove when dropping support for Python 3.14 (EOL: 10/2030)
PY314_OR_LESS: bool = (PY_VERS < (3, 15))

#############
# OS checks #
#############
OS_WINDOWS: bool = platform.system() == 'Windows'
"""Operating system is Windows"""

OS_LINUX: bool = platform.system() == 'Linux'
"""Operating system is Linux"""

OS_MACOS: bool = platform.system() == 'Darwin'
"""Operating system is macOS"""

OS_POSIX: bool = OS_LINUX or OS_MACOS
"""Operating system is POSIX compliant (Linux or macOS)"""