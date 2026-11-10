__all__ = ['Logger', '__date__', '__version__', '__version_hex__', '__version_tuple__', 'settings', 'version']

import threading
from typing import TYPE_CHECKING, Any

from .logging import Logger
from .config.version import version

if TYPE_CHECKING:
    from .config.settings import XTLSettings
    from xtl_core.version import VersionInfo

logger = Logger(__name__)
logger.info(f'XTL version: {version} [{version.release_date}]')

version: 'VersionInfo'
"""
Version information for XTL.
"""

__version__ = version.string
__version_tuple__ = version.tuple_safe
__version_hex__ = version.hex
__date__ = version.date

# NB: `settings` is only annotated, never assigned here, so that the module-level
#   `__getattr__` below is called on first access (PEP 562). Assigning it would
#   load the settings on import.
settings: 'XTLSettings'
"""
Shared settings across XTL, initialized from ``xtl.toml`` on first access.

:meta hide-value:
"""

_settings_lock = threading.RLock()
_settings_loading = False


def _load_settings() -> 'XTLSettings':
    """
    Initialize the shared settings once and cache them in the module namespace. Safe
    to call from multiple threads; raises if settings are accessed while they are
    being initialized (e.g. from a module imported by :mod:`xtl.config.settings`).
    """
    global _settings_loading
    with _settings_lock:
        # Another thread might have initialized the settings while we were waiting
        if 'settings' in globals():
            return globals()['settings']
        # The RLock lets the same thread re-enter, so recursion is caught here
        if _settings_loading:
            raise RuntimeError(
                'xtl.settings was accessed while it was being '
                'initialized. Import it lazily inside the function '
                'that needs it instead.'
            )
        _settings_loading = True
        try:
            from .config.settings import XTLSettings

            _settings = XTLSettings.initialize()
            globals()['settings'] = _settings
        finally:
            _settings_loading = False
        return _settings


def __getattr__(name: str) -> Any:
    if name == 'settings':
        return _load_settings()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
