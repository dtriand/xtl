__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'cfg'
]

from .version import version

__version__ = version.string_safe
__version_tuple__ = version.tuple_safe
__version_hex__ = version.hex
__date__ = version.date

from .config import cfg
