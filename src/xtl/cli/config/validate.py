from pathlib import Path
from typing import Optional

import typer

from xtl.cli.utilities import epilog


app = typer.Typer()


@app.command(
    name='validate',
    short_help='Check a config file for errors and warnings',
    epilog=epilog
)
def cli_config_validate(
    file: Optional[Path] = typer.Argument(
        None, metavar='[FILE]', exists=True, dir_okay=False, resolve_path=True,
        help='Config file to check (default: the active config file)'
    )
):
    """
    Check a config file for errors (the file cannot be read, invalid TOML or invalid
    values) and warnings (version mismatch, unknown keys).
    """
    from rich.console import Console
    from rich.markup import escape

    from xtl.config.settings import XTLSettings

    console = Console(highlight=False)

    # Default to the active config file
    if file is None:
        active = next((loc for loc in XTLSettings.locations() if loc.exists), None)
        if active is None:
            console.print('No config file found, nothing to validate. '
                          'Using [b]in-memory defaults[/b].')
            return
        console.print(f'Validating [b]{active.name}[/b] config: '
                      f'{escape(str(active.path))}')
        file = active.path
    else:
        console.print(f'Validating config: {escape(str(file))}')

    errors, warnings = XTLSettings.check_file(file)

    for error in errors:
        console.print(f'[red]Error:[/] {escape(error)}')
    for warning in warnings:
        console.print(f'[yellow]Warning:[/] {escape(warning)}')

    if errors:
        console.print('[red]File is invalid: XTL will fall back to in-memory '
                      'defaults when loading this file. [/]')
        raise typer.Exit(1)
    elif warnings:
        s = 's' if len(warnings) > 1 else ''
        console.print(f'[yellow]File is valid with {len(warnings)} warning{s}.[/]')
    else:
        console.print('[green]File is valid.[/]')
