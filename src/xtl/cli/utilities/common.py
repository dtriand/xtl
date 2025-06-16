from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import typer

from xtl import settings
from xtl.automate import ComputeSite
from xtl.common.os import FilePermissions

if TYPE_CHECKING:
    from xtl.jobs.config import BatchConfig


CPU_CORES: int = os.cpu_count()
"""Number of CPU cores available on the system, used for parallel job execution."""


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


# Modified by `xtl.cli.utilities.decorators.depends_on`
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
class AutomateOptions:
    compute_site: ComputeSite
    permissions: PermissionsOptions
    modules: set[str]
    keep_temp: bool

    def get_batch_config(self) -> 'BatchConfig':
        from xtl.jobs.config import BatchConfig

        compute_site = self.compute_site.get_site()
        if self.compute_site == 'modules':
            dependencies = self.modules
        else:
            dependencies = REQUIRED_DEPENDENCIES

        batch_config = BatchConfig(
            compute_site=compute_site,
            dependencies=dependencies
        )

        if self.compute_site == 'modules':
            batch_config._strict_resolution = False
        return batch_config


automate_options_panel = 'Job execution'
def get_automate_options(
    compute_site: ComputeSite = \
        typer.Option(
            settings.automate.compute_site, '--compute-site',
            rich_help_panel=automate_options_panel,
            help='Compute site for configuring job execution'),
    modules: list[str] = \
        typer.Option(
            REQUIRED_MODULES, '--module',
            rich_help_panel=automate_options_panel,
            help='Module to load before job execution (only for `modules` site)'),
    update_permissions: bool = \
        typer.Option(
            settings.automate.permissions.update,
            '--chmod/--dont-chmod',
            rich_help_panel=automate_options_panel,
            help='Update permissions of output files',
        ),
    permissions: tuple[FilePermissions, FilePermissions] = \
        typer.Option(
            (settings.automate.permissions.files,
             settings.automate.permissions.directories),
            '--permissions',
            rich_help_panel=automate_options_panel,
            metavar='<FILES DIRS>',
            parser=FilePermissions.from_string,
            help='Permissions for output files and directories'
        ),
    keep_temp: bool = \
        typer.Option(
            settings.automate.keep_temp, '--keep-temp/--delete-temp',
            rich_help_panel=automate_options_panel,
            help='Keep temporary files after job execution'
        )
):
    return AutomateOptions(
        compute_site=compute_site,
        permissions=PermissionsOptions(
            update=update_permissions,
            files=FilePermissions(permissions[0]),
            directories=FilePermissions(permissions[1])
        ),
        modules=set(modules),
        keep_temp=keep_temp
    )
