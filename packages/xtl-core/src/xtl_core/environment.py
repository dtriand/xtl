"""
Environment variables that are read on every access. Packages define their own
variables with :class:`EnvVar`; xtl's are in :mod:`xtl.shared.environment`.
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


def expand_path(value: str) -> Path:
    """
    Parse a path, expanding environment variables and ``~``; for use as
    :attr:`EnvVar.parser`.
    """
    return Path(os.path.expandvars(value)).expanduser()
