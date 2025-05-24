__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'settings'
]

from .config.version import version as _version


version: 'xtl.config.version.VersionInfo' = _version
"""
Version information for XTL.
"""

__version__ = version.string
__version_tuple__ = version.tuple_safe
__version_hex__ = version.hex
__date__ = version.date


from .config.settings import XTLSettings

settings: XTLSettings = XTLSettings.initialize()
"""
Shared settings across XTL, initialized from ``xtl.toml``.

:meta hide-value:
"""
