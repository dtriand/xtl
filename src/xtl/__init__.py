__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'settings', 'Logger'
]

from .logging import Logger
logger = Logger(__name__)

from .config.version import version
logger.info(f'XTL version: {version} [{version.release_date}]')

version: 'xtl.config.version.VersionInfo'
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
