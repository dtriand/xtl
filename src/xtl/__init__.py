__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'settings'
]

from .config.version import version


version: 'xtl.config.version.VersionInfo'
"""
Version information for XTL.
"""

__version__ = version.string
__version_tuple__ = version.tuple_safe
__version_hex__ = version.hex
__date__ = version.date


# Import guard for build tools
import sys
if any(tool in arg for tool in ['setuptools', 'pip', 'egg_info', 'bdist_wheel']
       for arg in sys.argv):
    # Skip imports if running any build tools because no dependencies are available
    pass
else:
    from .config.settings import XTLSettings

    settings: XTLSettings = XTLSettings.initialize()
    """
    Shared settings across XTL, initialized from ``xtl.toml``.

    :meta hide-value:
    """