__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'cfg'
]

from .version import version

__version__ = version.string
__version_tuple__ = version.safe_tuple
__version_hex__ = version.hex
__date__ = version.date

from .config import cfg
