import logging

from xtl_core._version import __version__

__all__ = ['__version__']

# Libraries only add a `NullHandler`; applications configure handlers, levels and formats
logging.getLogger(__name__).addHandler(logging.NullHandler())
