from dataclasses import dataclass

import typer

from xtl import settings
from xtl.common.os import FilePermissions, CPU_CORES
from xtl.jobs.sites import ComputeSite


@dataclass(frozen=True)
class ConsoleOptions:
    verbose: int
    debug: bool


console_options_panel = 'Debugging'
def get_console_options(
    verbose: int = \
        typer.Option(
            0, '--verbose', '-v',
            count=True,
            rich_help_panel=console_options_panel,
            help='Print additional information'),
    debug: bool = \
        typer.Option(
            False, '--debug',
            show_default=True,
            rich_help_panel=console_options_panel,
            help='Print debug information'),
):
    return ConsoleOptions(verbose=verbose, debug=debug)


# Modified by `xtl.cli.utilities.decorators.job_options`
REQUIRED_DEPENDENCIES: dict[str, list[str]] = {'extra': []}
"""Dictionary of required dependencies and their modules for job execution, modified by decorators."""

REQUIRED_MODULES: list[str] = list()
"""List of required modules for job execution, modified by decorators."""


@dataclass(frozen=True)
class PermissionsOptions:
    update: bool
    files: FilePermissions
    directories: FilePermissions

    @staticmethod
    def parse_permissions_pair(value: str) -> tuple[FilePermissions, FilePermissions]:
        parts = [p.strip() for p in value.split(',')]
        if len(parts) != 2 or not all(parts):
            raise typer.BadParameter('Expected format: FILES,DIRS (example: 600,700)')

        try:
            files = FilePermissions.from_string(parts[0]) if parts[0] else settings.jobs.permissions.files
            dirs = FilePermissions.from_string(parts[1]) if parts[1] else settings.jobs.permissions.directories
        except Exception as e:
            raise typer.BadParameter('Invalid permissions format') from e
        return files, dirs


@dataclass(frozen=True)
class JobOptions:
    compute_site: ComputeSite
    permissions: PermissionsOptions
    modules: dict[str, list[str]]
    keep_temp: bool

    @staticmethod
    def _ignore_default_modules(value: str | list[str]) -> str:
        if isinstance(value, list):
            return ''
        elif isinstance(value, str):
            return value
        else:
            raise ValueError(f'Could not cast {value!r} to string')

    @staticmethod
    def parse_module_names(value: str) -> dict[str, list[str]]:
        modules = {
            'extra': []
        }

        if not value:
            return modules

        module_parts = value.split(',')
        for part in module_parts:
            dependency_parts = part.split('=')
            if len(dependency_parts) == 1:
                modules['extra'].append(dependency_parts[0])
            elif len(dependency_parts) == 2:
                dep, module = dependency_parts
                if dep not in modules.keys():
                    modules[dep] = []
                modules[dep].append(module)
            else:
                raise typer.BadParameter(f'Invalid module format: {part!r}. Expected <module> or <dependency>=<module>')
        return modules

    @property
    def has_modules(self) -> bool:
        return any(self.modules.values())

job_options_panel = 'Job execution'
def get_job_options(
    compute_site: ComputeSite = \
        typer.Option(
            settings.jobs.compute_site, '--compute-site',
            rich_help_panel=job_options_panel,
            help='Compute site for configuring job execution'),
    modules: str = \
        typer.Option(
            REQUIRED_MODULES,
            '--modules',
            parser=JobOptions._ignore_default_modules,
            rich_help_panel=job_options_panel,
            help='Module to load before job execution (only for `modules` site)'),
    update_permissions: bool = \
        typer.Option(
            settings.jobs.permissions.update,
            '--chmod/--dont-chmod',
            rich_help_panel=job_options_panel,
            help='Update permissions of output files',
        ),
    permissions: str = \
        typer.Option(
            f'{settings.jobs.permissions.files}, {settings.jobs.permissions.directories}',
            '--permissions',
            rich_help_panel=job_options_panel,
            metavar='FILES,DIRS',
            help='Permissions for output files and directories'
        ),
    keep_temp: bool = \
        typer.Option(
            settings.jobs.keep_temp, '--keep-temp/--delete-temp',
            rich_help_panel=job_options_panel,
            help='Keep temporary files after job execution'
        )
):
    f_permissions, d_permissions = PermissionsOptions.parse_permissions_pair(permissions)
    return JobOptions(
        compute_site=compute_site,
        permissions=PermissionsOptions(
            update=update_permissions,
            files=f_permissions,
            directories=d_permissions
        ),
        modules=JobOptions.parse_module_names(modules),
        keep_temp=keep_temp
    )
