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
            raise ValueError(f'Invalid value for environment variable {self.name}: {raw!r}') from e


def _normalize_path(value: str) -> Path:
    """
    Normalize a path string by expanding environment variables and user home directory.
    """
    return Path(os.path.expandvars(value)).expanduser()


#########################
# Environment variables #
#########################
XTL_CONFIG: EnvVar[Path] = EnvVar('XTL_CONFIG', parser=_normalize_path)
"""
Path to the config file, which takes precedence over all other config file locations
(see :meth:`XTLSettings.locations() <xtl.config.settings.XTLSettings.locations>`).
"""

XTL_COMPUTE_SITE: EnvVar[str] = EnvVar('XTL_COMPUTE_SITE')
"""
Enables compute sites (see :class:`xtl.jobs.sites.ComputeSite`) for specific facilities.
Read when :mod:`xtl.jobs.sites` is imported. Currently supported values are:

- ``BIOTIX_HPC`` enables ``biotix``
"""

XTL_FONTS_DIR: EnvVar[Path] = EnvVar('XTL_FONTS_DIR', parser=_normalize_path)
"""
Directory with additional fonts for plots. Reserved for the plotting backends, which
are still in development; it has no effect yet.
"""
