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
import logging
from typing import Any, Iterable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from xtl.config.settings import DependencySettings
    from xtl.jobs.shells import Shell, ShellType
    from xtl.jobs.sites import ComputeSite, ComputeSiteType


__all__ = [
    'DependencyCheck',
    'dependency_name',
    'unknown_dependencies',
    'site_key',
    'select_site_and_shell',
    'ensure_dependencies',
]

logger = logging.getLogger(__name__)


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


def _agreed(dependencies: list[DependencySettings], field: str) -> Any:
    """
    The value of a field that the dependencies set (``compute_site``, ``shell`` or ``policy``), or None if none does.

    :raises DependencyError: If they set different values.
    """
    from xtl.jobs.exceptions import DependencyError

    # NB: Settings aren't hashable, so pairs rather than a dict
    values = [(dep, getattr(dep, field)) for dep in dependencies if getattr(dep, field) is not None]
    if len({value for _, value in values}) > 1:
        names = tuple(dependency_name(dep) for dep, _ in values)
        asked = ', '.join(f'{dependency_name(dep)}: {value}' for dep, value in values)
        raise DependencyError(f'The dependencies ask for different values of `{field}` ({asked})', dependencies=names)
    return values[0][1] if values else None


def select_site_and_shell(
    dependencies: Iterable[DependencySettings | str] | None = None,
    compute_site: Optional[ComputeSite | str] = None,
    default_shell: Optional[Shell | str] = None,
    shells: Optional[Iterable[Shell]] = None,
) -> tuple[ComputeSiteType, ShellType]:
    """
    The compute site and shell that run a job with these dependencies.

    The compute site is ``compute_site`` if given, else the one the dependencies set in their settings, else
    ``jobs.compute_site``; its policy is the one the dependencies set, else its own. The shell is ``default_shell`` if
    given, else the one the dependencies set, else the site's default, else the system's, as long as both the site and
    ``shells`` (*e.g.* those a batch job has templates for) allow it; otherwise one they both allow. If they allow
    none in common, the system's shell, with a warning.

    :param dependencies: Names or settings of the job's dependencies.
    :param compute_site: The compute site the job's configuration asks for, if any.
    :param default_shell: The shell the job's configuration asks for, if any.
    :param shells: The shells the job can run in, if it is limited to some.
    :raises DependencyError: If the dependencies ask for different compute sites, shells or policies.
    :raises ValueError: If the shell asked for (by the configuration or the dependencies) isn't one the site and
        ``shells`` allow.
    """
    from xtl import settings
    from xtl.jobs.shells import DefaultShell, Shell
    from xtl.jobs.sites import ComputeSite

    def as_shell(shell: Any) -> Shell:
        # Enum members, from members, names or instances; the default shell as the system's
        shell = Shell(shell.name) if not isinstance(shell, str) else Shell(shell)
        return Shell(DefaultShell().name) if shell == Shell.DEFAULT else shell

    resolved = settings.dependencies.resolve(dependencies)

    # Compute site, with the policy of the dependencies
    name = compute_site if compute_site is not None else _agreed(resolved, 'compute_site')
    site = ComputeSite(name if name is not None else settings.jobs.compute_site).get(policy=_agreed(resolved, 'policy'))

    # Shells that both the site and the job allow (None: any)
    allowed: Optional[set[Shell]] = None
    for limit in (site.supported_shells, shells):
        if limit:
            limit = {as_shell(shell) for shell in limit}
            allowed = limit if allowed is None else allowed & limit

    system = as_shell(Shell.DEFAULT)
    asked = default_shell if default_shell is not None else _agreed(resolved, 'shell')
    if asked is not None:
        asked = as_shell(asked)
        if allowed is not None and asked not in allowed:
            raise ValueError(
                f'Shell {asked.value} is not one that the compute site {name or settings.jobs.compute_site} and the '
                f'job allow ({", ".join(sorted(shell.value for shell in allowed)) or "none"})'
            )
        return site, asked.get()

    preferred = [as_shell(site.default_shell) if site.default_shell is not None else None, system]
    for shell in preferred:
        if shell is not None and (allowed is None or shell in allowed):
            return site, shell.get()
    if allowed:
        return site, min(allowed, key=lambda shell: shell.value).get()
    logger.warning('No shell that both compute site %s and the job allow; using %s', type(site).__name__, system.value)
    return site, system.get()


def ensure_dependencies(
    dependencies: Iterable[DependencySettings | str] | None, site: ComputeSiteType, shell: ShellType
) -> None:
    """
    Make sure that dependencies can be found on a compute site, in a shell, as checked by the site (see
    :meth:`~xtl.jobs.sites.LocalSite.check_dependencies`): each is checked once, then the outcome is remembered in its
    settings. Nothing is checked if ``settings.dependencies.check`` is off.

    :param dependencies: Names or settings of the dependencies.
    :param site: The compute site.
    :param shell: The shell.
    :raises DependencyError: If a dependency is missing, with what the check found.
    """
    from xtl import settings
    from xtl.jobs.exceptions import DependencyError

    if not dependencies or not settings.dependencies.check:
        return
    missing = [check for check in site.check_dependencies(dependencies, shell=shell) if not check.found]
    if not missing:
        return
    lines = [
        f'Missing dependenc{"y" if len(missing) == 1 else "ies"} on {type(site).__name__} in {shell.name}:',
        *(f'- {check.dependency}: {", ".join(check.missing) or "not found"}' for check in missing),
    ]
    details = '\n'.join(check.details for check in missing if check.details)
    if details:
        lines += ['What the check printed:', details]
    raise DependencyError('\n'.join(lines), dependencies=tuple(check.dependency for check in missing))
