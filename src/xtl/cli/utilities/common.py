from dataclasses import dataclass
import os

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
REQUIRED_DEPENDENCIES = list()
"""List of required dependencies for job execution, modified by decorators."""

REQUIRED_MODULES = list()
"""List of required modules for job execution, modified by decorators."""


@dataclass(frozen=True)
class PermissionsOptions:
    update: bool
    files: FilePermissions
    directories: FilePermissions


@dataclass(frozen=True)
class JobOptions:
    compute_site: ComputeSite
    permissions: PermissionsOptions
    modules: set[str]
    keep_temp: bool


job_options_panel = 'Job execution'
def get_job_options(
    compute_site: ComputeSite = \
        typer.Option(
            settings.jobs.compute_site, '--compute-site',
            rich_help_panel=job_options_panel,
            help='Compute site for configuring job execution'),
    modules: list[str] = \
        typer.Option(
            REQUIRED_MODULES, '--module',
            rich_help_panel=job_options_panel,
            help='Module to load before job execution (only for `modules` site)'),
    update_permissions: bool = \
        typer.Option(
            settings.automate.permissions.update,
            '--chmod/--dont-chmod',
            rich_help_panel=job_options_panel,
            help='Update permissions of output files',
        ),
    permissions: tuple[FilePermissions, FilePermissions] = \
        typer.Option(  # TODO: Check parser, should accept one comma separated argument, e.g.: 700,600
            (settings.automate.permissions.files,
             settings.automate.permissions.directories),
            '--permissions',
            rich_help_panel=job_options_panel,
            metavar='<FILES DIRS>',
            parser=FilePermissions.from_string,
            help='Permissions for output files and directories'
        ),
    keep_temp: bool = \
        typer.Option(
            settings.automate.keep_temp, '--keep-temp/--delete-temp',
            rich_help_panel=job_options_panel,
            help='Keep temporary files after job execution'
        )
):
    return JobOptions(
        compute_site=compute_site,
        permissions=PermissionsOptions(
            update=update_permissions,
            files=FilePermissions(permissions[0]),
            directories=FilePermissions(permissions[1])
        ),
        modules=set(modules),
        keep_temp=keep_temp
    )
