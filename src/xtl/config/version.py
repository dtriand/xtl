"""
The one place where xtl's version is defined.

Hatch reads the version string with a regex when building the package (see
``[tool.hatch.version]`` in ``pyproject.toml``), without running this module, so keep it
as the first argument of ``VersionInfo.from_str()``, in single quotes.
"""

from xtl_core.version import VersionInfo

version = VersionInfo.from_str('0.2.0dev0', date_str='01/06/2025')  # serial < 16
"""XTL version information"""
