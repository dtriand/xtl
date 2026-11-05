from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from xtl.cli.utilities import epilog


app = typer.Typer()


class Location(str, Enum):
    ENV = 'env'
    LOCAL = 'local'
    USER = 'user'
    GLOBAL = 'global'


def _target_path(name: str) -> Path:
    """
    Path of the config file for a given location name.
    """
    from xtl.config.settings import XTLSettings
    from xtl.common.environment import XTL_CONFIG

    if name == 'env':
        # locations() drops XTL_CONFIG if it is unset or a directory
        if XTL_CONFIG.raw is None:
            raise typer.BadParameter('XTL_CONFIG is not set', param_hint='--env')
        if XTL_CONFIG.value.is_dir():
            raise typer.BadParameter(f'XTL_CONFIG points to a directory: {XTL_CONFIG.value}', param_hint='--env')
        return XTL_CONFIG.value
    return next(loc.path for loc in XTLSettings.locations() if loc.name == name)


@app.command(name='create', short_help='Create a config file', epilog=epilog)
def cli_config_create(
    location: Optional[Location] = typer.Option(
        None, '--location', '-l', case_sensitive=False, help='Location of the new config file'
    ),
    path: Optional[Path] = typer.Option(
        None, '--path', '-p', dir_okay=False, resolve_path=True, help='Create the config file at a custom path'
    ),
    clean: bool = typer.Option(False, '--clean', help='Write a clean config file with default settings'),
    force: bool = typer.Option(False, '--force', '-f', help='Overwrite the file if it already exists'),
    env: bool = typer.Option(False, '--env', rich_help_panel='Aliases', help='Alias for --location env'),
    local: bool = typer.Option(False, '--local', rich_help_panel='Aliases', help='Alias for --location local'),
    user: bool = typer.Option(False, '--user', rich_help_panel='Aliases', help='Alias for --location user'),
    global_: bool = typer.Option(False, '--global', rich_help_panel='Aliases', help='Alias for --location global'),
):
    """
    Create a config file in the current directory (default) or a different location.
    The new file is a copy of the active config file or the default settings if none
    exists.
    """
    from rich.console import Console
    from rich.markup import escape

    from xtl.config.settings import XTLSettings

    console = Console(highlight=False)

    # Resolve the target location (--location and its shortcuts are mutually exclusive)
    chosen = {name for name, flag in (('env', env), ('local', local), ('user', user), ('global', global_)) if flag}
    if location is not None:
        chosen.add(location.value)
    if path is not None:
        chosen.add('path')
    if len(chosen) > 1:
        raise typer.BadParameter('--location, --env, --local, --user, --global and --path are mutually exclusive')
    name = chosen.pop() if chosen else 'local'
    target = path if name == 'path' else _target_path(name)

    # Pick the settings to write
    if clean:
        settings = XTLSettings()
        origin = 'the default settings'
    else:
        active = XTLSettings.active()
        if active is None:
            settings = XTLSettings()
            origin = 'the default settings (no config file found)'
        else:
            settings, errors, _ = XTLSettings._read(active.path)
            if settings is None:
                console.print(
                    f'[red]Error:[/] the active [b]{active.name}[/b] config is invalid: {escape("; ".join(errors))}'
                )
                console.print(
                    'Fix it (see [b]xtl.config validate[/b]) or use [b]--clean[/b] to write the default settings.'
                )
                raise typer.Exit(1)
            origin = f'the {active.name} config ({escape(str(active.path))})'

    # Write the new config file
    if target.exists() and not force:
        console.print(f'[red]Error:[/] file already exists: {escape(str(target))}. Use [b]--force[/b] to overwrite it.')
        raise typer.Exit(1)
    try:
        settings.to_toml(target, comments=True, overwrite=force, keep_file_ext=True)
    except OSError as e:
        console.print(f'[red]Error:[/] failed to write {escape(str(target))}: {escape(str(e))}')
        raise typer.Exit(1)
    console.print(f'Created config at {escape(str(target))} from {origin}')

    # Tell whether the new file will actually be used
    locations = XTLSettings.locations()
    active = XTLSettings.active(locations)
    if not any(loc.exists and loc.path.samefile(target) for loc in locations):
        console.print('[dim]This file is not in the lookup path; set XTL_CONFIG to use it.[/]')
    elif not active.path.samefile(target):
        console.print(
            f'[yellow]Note:[/] the [b]{active.name}[/b] config at '
            f'{escape(str(active.path))} takes precedence over this file.'
        )
