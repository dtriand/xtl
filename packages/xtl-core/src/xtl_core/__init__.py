import logging

from xtl_core.version_info import VersionInfo

__all__ = ['__version__', 'version']

# NB: Hatch reads the version string with a regex when building the package (see `[tool.hatch.version]` in
#  `pyproject.toml`), without running this module, so keep it as the first argument of `VersionInfo.from_str()`, in
#  single quotes, in its PEP 440 form (hatch writes it into the package metadata as is)
version = VersionInfo.from_str('0.1.0.dev0')  # serial < 16
"""xtl-core version information"""

__version__ = version.pep440

# Libraries only add a `NullHandler`; applications configure handlers, levels and formats
logging.getLogger(__name__).addHandler(logging.NullHandler())
