"""
Central location for environment variables and related utilities.
"""
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar('T')


@dataclass(frozen=True)
class EnvVar(Generic[T]):
    """
    An environment variable that is read on every access, so that changes to the
    environment are reflected immediately. Empty values are treated as unset.
    """
    name: str
    parser: Callable[[str], T] = str
    default: T | None = None

    @property
    def raw(self) -> str | None:
        """The raw string value of the environment variable, or None if unset."""
        return os.environ.get(self.name) or None

    @property
    def is_set(self) -> bool:
        """True if the environment variable is set, False otherwise."""
        return self.raw is not None

    @property
    def value(self) -> T | None:
        """The current value parsed with :attr:`parser`, or the default if unset."""
        raw = self.raw
        if raw is None:
            return self.default
        try:
            return self.parser(raw)
        except (ValueError, TypeError) as e:
            raise ValueError(f'Invalid value for environment variable '
                             f'{self.name}: {raw!r}') from e


def _normalize_path(value: str) -> Path:
    """
    Normalize a path string by expanding environment variables and user home directory.
    """
    return Path(os.path.expandvars(value)).expanduser()


#########################
# Environment variables #
#########################
XTL_CONFIG: EnvVar[Path] = EnvVar('XTL_CONFIG', parser=_normalize_path)
"""Environment variable 'XTL_CONFIG'"""
# Used to override the default settings file path

XTL_COMPUTE_SITE: EnvVar[str] = EnvVar('XTL_COMPUTE_SITE')
"""Environment variable 'XTL_COMPUTE_SITE'"""
# Used to trigger specialized compute site configurations

XTL_FONTS_DIR: EnvVar[Path] = EnvVar('XTL_FONTS_DIR', parser=_normalize_path)
"""Environment variable 'XTL_FONTS_DIR'"""
# Used to load fonts from custom directories
