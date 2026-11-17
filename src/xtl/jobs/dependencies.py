"""
Dependencies of jobs on external programs.

A dependency is declared in ``settings.dependencies``: where the program is (``path``, ``modules``), and optionally the
compute site, shell and command policy it runs with. Jobs name the dependencies they need, and compute sites set them
up and check that they can be found. The settings of each dependency also remember its checks, so that a command, a
pool and its jobs check it once for a given compute site and shell, and the workers of a process pool receive them with
the settings. This module holds what jobs, sites and the settings share: the names the settings know, the outcome of a
check, and how a compute site is told apart in the records of checks.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings


__all__ = [
    'DependencyCheck',
    'dependency_name',
    'unknown_dependencies',
    'site_key',
]


@dataclass(frozen=True)
class DependencyCheck:
    """
    The outcome of checking a dependency on a compute site, in a shell.
    """

    dependency: str
    """The name of the dependency, e.g. ``'atsas'``"""

    found: bool
    """Whether everything the dependency provides was found"""

    missing: tuple[str, ...] = ()
    """What wasn't found: modules that didn't load, executables that weren't there"""

    details: str = ''
    """What the check printed, to show how it looked"""

    def to_dict(self) -> dict[str, Any]:
        """
        The check as plain values, e.g. to send it to another process.
        """
        return {
            'dependency': self.dependency,
            'found': self.found,
            'missing': list(self.missing),
            'details': self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DependencyCheck:
        """
        A check from the output of :meth:`to_dict`.

        :param data: The check as plain values.
        """
        return cls(
            dependency=data['dependency'],
            found=data['found'],
            missing=tuple(data.get('missing', ())),
            details=data.get('details', ''),
        )


def dependency_name(dependency: DependencySettings) -> str:
    """
    The name the settings know a dependency by, e.g. ``'atsas'``, or its class name if they don't.

    :param dependency: The settings of a dependency.
    """
    from xtl import settings

    return next(
        (name for name, known in settings.dependencies.items() if known is dependency), type(dependency).__name__
    )


def unknown_dependencies(names: Iterable[str]) -> list[str]:
    """
    The names that ``settings.dependencies`` doesn't know, sorted, e.g. to test the dependencies a job declares.

    :param names: Names of dependencies.
    """
    from xtl import settings

    return sorted({name for name in names if name not in settings.dependencies})


def site_key(site: Any) -> str:
    """
    A compute site as text, to tell its checks apart from those of other sites: its class, the shells it supports and
    its policy.

    :param site: A compute site.
    """
    shells = ','.join(sorted(str(shell) for shell in site.supported_shells))
    policy = type(site.policy).__name__ if site.policy is not None else ''
    return f'{type(site).__qualname__}[{shells}][{policy}]'
