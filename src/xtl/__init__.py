__all__ = [
    'version', '__version__', '__version_tuple__', '__version_hex__', '__date__',
    'cfg'
]

from .version import version

__version__ = version.string
__version_tuple__ = version.tuple_safe
__version_hex__ = version.hex
__date__ = version.date

# Workaround for missing package when checking for __version__ during building
try:
    from .config import cfg
except ModuleNotFoundError as e:
    import sys
    if not any(arg.startswith('setuptools') for arg in sys.argv):
        pass
    else:
        raise e
